"""Profile Base and Set C3DQN-NCDM checkpoints on one shared Top-K batch."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agents.ncdm_c3dqn_trainer_v2 import (
    forward_q_network,
    load_c3dqn_checkpoint,
    load_set_c3dqn_checkpoint,
)
from models.ncdm import OfficialNCDM, load_q_matrix, safe_load_ncdm_checkpoint
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    return dict(checkpoint.get("metadata") or {})


def _assert_shared_protocol(
    base_metadata: dict[str, Any],
    set_metadata: dict[str, Any],
    *,
    top_k: int,
) -> int:
    fields = [
        "knowledge_dim",
        "selection_horizon",
        "warm_start_items",
        "q_matrix_item_count",
        "ncdm_item_count",
        "alpha_fit",
        "reward_config",
    ]
    differences = {
        field: (base_metadata.get(field), set_metadata.get(field))
        for field in fields
        if base_metadata.get(field) != set_metadata.get(field)
    }
    if differences:
        raise ValueError(f"Base/Set checkpoint protocol mismatch: {differences}")

    for label, metadata in (("Base", base_metadata), ("Set", set_metadata)):
        pool = dict(metadata.get("candidate_pool_config") or {})
        if not bool(pool.get("prefilter_enabled", False)):
            raise ValueError(f"{label} checkpoint did not enable candidate prefilter")
        actual_top_k = int(pool.get("prefilter_top_k", -1))
        if actual_top_k != int(top_k):
            raise ValueError(
                f"{label} checkpoint Top-K mismatch: {actual_top_k} != {top_k}"
            )
    return int(base_metadata["selection_horizon"])


def _build_shared_batch(
    *,
    cache: NCDMItemFeatureCache,
    ncdm: OfficialNCDM,
    q_matrix: torch.Tensor,
    candidate_pool_config: dict[str, Any],
    top_k: int,
    raw_candidate_count: int,
    batch_size: int,
    history_length: int,
    selection_horizon: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    available = cache.item_count - history_length
    if available < top_k:
        raise ValueError(
            f"not enough cached items for Top-K={top_k}: available={available}"
        )
    raw_count = min(max(top_k, raw_candidate_count), available)
    history_items = list(range(history_length))
    history_responses = [float(index % 2 == 0) for index in range(history_length)]
    raw_candidates = list(range(history_length, history_length + raw_count))
    coverage_count = cache.q_masks[
        torch.tensor(history_items, dtype=torch.long, device=cache.device)
    ].sum(dim=0)
    alpha = torch.zeros((1, cache.knowledge_dim), device=cache.device)
    mastery = torch.sigmoid(alpha).squeeze(0)
    prefilter = NCDMCandidatePrefilter(
        q_matrix=q_matrix,
        feature_cache=cache,
        ncdm=ncdm,
        config=candidate_pool_config,
    )
    filtered, summary = prefilter.select(
        raw_candidates,
        alpha,
        mastery,
        coverage_count,
    )
    if len(filtered) != min(top_k, raw_count):
        raise ValueError(
            f"shared prefilter returned {len(filtered)} candidates, expected {top_k}"
        )

    rows = []
    for row_index in range(batch_size):
        responses = [
            float((index + row_index) % 2 == 0)
            for index in range(history_length)
        ]
        rows.append(
            {
                "history_item_ids": history_items,
                "history_responses": responses,
                "candidate_item_ids": filtered,
                "selected_item_id": filtered[row_index % len(filtered)],
                "mastery": mastery.detach().cpu().tolist(),
                "coverage_count": coverage_count.detach().cpu().tolist(),
                "coverage": (
                    coverage_count / float(max(1, selection_horizon))
                ).clamp(0, 1).detach().cpu().tolist(),
                "policy_step": min(history_length - 1, selection_horizon - 1),
            }
        )
    batch = pad_c3dqn_batch(
        rows,
        cache,
        selection_horizon,
        require_exact_coverage=True,
    )
    return batch, {
        **summary,
        "shared_candidate_ids": filtered,
        "shared_candidate_count": len(filtered),
        "raw_candidate_count": raw_count,
    }


def _profile_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    chunk_size: int | None,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward_q_network(model, batch, chunk_size=chunk_size)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        else:
            baseline = 0
        timings = []
        for _ in range(repeats):
            _sync(device)
            start = time.perf_counter()
            q_values, _ = forward_q_network(model, batch, chunk_size=chunk_size)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        if not torch.isfinite(q_values).all():
            raise ValueError("non-finite Q values during forward profile")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = 0
    return {
        "forward_ms_mean": statistics.mean(timings),
        "forward_ms_std": statistics.pstdev(timings),
        "forward_peak_total_mb": peak / 1024.0**2,
        "forward_peak_incremental_mb": max(0, peak - baseline) / 1024.0**2,
    }


def _profile_update(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    chunk_size: int | None,
    use_amp: bool,
) -> dict[str, float]:
    train_model = copy.deepcopy(model).to(device).train()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=1.0e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True)
            if use_amp
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with autocast_context:
            q_values, _ = forward_q_network(
                train_model,
                batch,
                chunk_size=chunk_size,
            )
            valid_q = q_values[batch["candidate_mask"]]
            target = torch.zeros_like(valid_q)
            loss = F.smooth_l1_loss(valid_q, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach().item())

    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    timings = []
    loss_value = float("nan")
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        loss_value = step()
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    if not torch.isfinite(torch.tensor(loss_value)):
        raise ValueError("non-finite loss during update profile")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0
    del train_model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "update_ms_mean": statistics.mean(timings),
        "update_ms_std": statistics.pstdev(timings),
        "update_peak_total_mb": peak / 1024.0**2,
        "update_peak_incremental_mb": max(0, peak - baseline) / 1024.0**2,
        "final_profile_loss": loss_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile paired Base/Set C3DQN checkpoints at shared Top-K"
    )
    parser.add_argument("--q-matrix", required=True)
    parser.add_argument("--ncdm-checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--set-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--raw-candidates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    q_matrix = load_q_matrix(args.q_matrix, device)
    ncdm = OfficialNCDM(1, q_matrix.shape[0], q_matrix.shape[1]).to(device)
    safe_load_ncdm_checkpoint(ncdm, args.ncdm_checkpoint, device)
    ncdm.eval()
    for parameter in ncdm.parameters():
        parameter.requires_grad_(False)
    cache = NCDMItemFeatureCache(ncdm, q_matrix, device)

    base_metadata = _checkpoint_metadata(args.base_checkpoint)
    set_metadata = _checkpoint_metadata(args.set_checkpoint)
    selection_horizon = _assert_shared_protocol(
        base_metadata,
        set_metadata,
        top_k=args.top_k,
    )
    base_model, _ = load_c3dqn_checkpoint(
        args.base_checkpoint,
        ncdm=ncdm,
        q_matrix=q_matrix,
        device=device,
    )
    set_model, _ = load_set_c3dqn_checkpoint(
        args.set_checkpoint,
        ncdm=ncdm,
        q_matrix=q_matrix,
        device=device,
    )
    batch, prefilter_summary = _build_shared_batch(
        cache=cache,
        ncdm=ncdm,
        q_matrix=q_matrix,
        candidate_pool_config=dict(
            base_metadata.get("candidate_pool_config") or {}
        ),
        top_k=args.top_k,
        raw_candidate_count=args.raw_candidates,
        batch_size=args.batch_size,
        history_length=args.history_length,
        selection_horizon=selection_horizon,
    )

    results = []
    for label, model, chunk_size in (
        ("Base-C3DQN-NCDM", base_model, None),
        ("Set-C3DQN-NCDM", set_model, args.chunk_size),
    ):
        row: dict[str, Any] = {
            "variant": label,
            "device": str(device),
            "amp": use_amp,
            "parameter_count": _parameter_count(model),
            "batch_size": args.batch_size,
            "history_length": args.history_length,
            "raw_candidate_count": prefilter_summary["raw_candidate_count"],
            "filtered_candidate_count": prefilter_summary[
                "shared_candidate_count"
            ],
            "top_k": args.top_k,
            "chunk_size": chunk_size or 0,
            "selection_horizon": selection_horizon,
        }
        row.update(
            _profile_forward(
                model,
                batch,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                chunk_size=chunk_size,
            )
        )
        row.update(
            _profile_update(
                model,
                batch,
                device=device,
                warmup=max(2, args.warmup // 2),
                repeats=args.repeats,
                chunk_size=chunk_size,
                use_amp=use_amp,
            )
        )
        results.append(row)
        print(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "results": results,
                "prefilter_summary": prefilter_summary,
            },
            indent=2,
        )
    )
    print(output_path)


if __name__ == "__main__":
    main()
