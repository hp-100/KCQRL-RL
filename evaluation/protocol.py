"""Deterministic support/query split protocol for benchmark_v2."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence


@dataclass
class StudentSplit:
    student_id: str
    support_item_ids: list[int]
    support_responses: list[float]
    query_item_ids: list[int]
    query_responses: list[float]
    warm_start_item: int
    warm_start_response: float
    seed: int
    original_interactions: int = 0
    valid_interactions: int = 0
    duplicate_interactions_removed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def valid_item_count(q_matrix, item_bank=None, ncdm=None, mirt=None, *, track: str | None = None) -> int:
    if track == "mirt_native":
        if mirt is None:
            raise ValueError("mirt_native valid_item_count requires a MIRT model")
        return int(min(mirt.disc_emb.num_embeddings, mirt.diff_emb.num_embeddings))
    counts = [len(q_matrix)]
    if item_bank is not None:
        counts.append(len(item_bank))
    if ncdm is not None:
        counts.extend([ncdm.k_difficulty.num_embeddings, ncdm.e_discrimination.num_embeddings])
    if mirt is not None:
        counts.extend([mirt.disc_emb.num_embeddings, mirt.diff_emb.num_embeddings])
    return min(int(c) for c in counts)


def student_rng(student_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{student_id}:{seed}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def clean_interactions_with_stats(item_ids: Sequence[int], responses: Sequence[float], max_item_id: int) -> tuple[list[int], list[float], int, int, int]:
    original = min(len(item_ids), len(responses))
    cleaned_i, cleaned_r = [], []
    seen: set[int] = set()
    duplicates = 0
    valid_before_dedupe = 0
    for item, resp in zip(item_ids, responses):
        try:
            item = int(item)
            resp = float(resp)
        except (TypeError, ValueError):
            continue
        if 0 <= item < max_item_id and resp == resp:
            valid_before_dedupe += 1
            if item in seen:
                duplicates += 1
                continue
            seen.add(item)
            cleaned_i.append(item)
            cleaned_r.append(1.0 if resp >= 0.5 else 0.0)
    return cleaned_i, cleaned_r, original, valid_before_dedupe, duplicates


def clean_interactions(item_ids: Sequence[int], responses: Sequence[float], max_item_id: int) -> tuple[list[int], list[float]]:
    items, responses, *_ = clean_interactions_with_stats(item_ids, responses, max_item_id)
    return items, responses


def make_student_split(student_id: str, item_ids: Sequence[int], responses: Sequence[float], *, seed: int, valid_count: int, query_ratio: float = 0.2, min_query_items: int = 5) -> tuple[StudentSplit | None, str | None]:
    items, resps, original_interactions, valid_interactions, duplicate_interactions_removed = clean_interactions_with_stats(item_ids, responses, valid_count)
    paired = list(zip(items, resps))
    if len(paired) < min_query_items + 2:
        return None, "too_few_valid_interactions"
    rng = student_rng(student_id, seed)
    order = list(range(len(paired)))
    rng.shuffle(order)
    query_n = max(min_query_items, int(round(len(paired) * query_ratio)))
    if len(paired) - query_n < 2:
        return None, "too_few_support_items"
    query_idx = set(order[:query_n])
    support = [paired[i] for i in order[query_n:]]
    query = [paired[i] for i in order[:query_n]]
    if not support or not query:
        return None, "empty_split"
    assert set(i for i, _ in support).isdisjoint(set(i for i, _ in query)), "support/query item leakage"
    warm_item, warm_resp = support[0]
    return StudentSplit(
        student_id=str(student_id),
        support_item_ids=[int(i) for i, _ in support],
        support_responses=[float(r) for _, r in support],
        query_item_ids=[int(i) for i, _ in query],
        query_responses=[float(r) for _, r in query],
        warm_start_item=int(warm_item),
        warm_start_response=float(warm_resp),
        seed=int(seed),
        original_interactions=int(original_interactions),
        valid_interactions=int(valid_interactions),
        duplicate_interactions_removed=int(duplicate_interactions_removed),
    ), None


def save_manifest(path: str | Path, splits: Sequence[StudentSplit], skipped: dict[str, str], config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "splits": [s.to_dict() for s in splits], "skipped": skipped}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
