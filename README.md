# KCQRL-RL

A minimal, runnable Knowledge-Concept Question Recommendation Learning (KCQRL-RL) framework for offline computerized adaptive testing (CAT) and DDPG policy training/evaluation.

## Repository layout

The project is intended to run from the repository root. Core folders live directly under the root:

```text
agents/ configs/ core/ datasets/ docs/ envs/ evaluation/ legacy/ models/ reward/ scripts/ utils/
```

Large data assets and checkpoints are **not** committed. See `docs/ASSET_MANIFEST.md` and configure all external data paths in `configs/default.yaml`.

## Colab workflow

1. Mount Google Drive in a Colab cell so `configs/default.yaml` can resolve the configured real data paths:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Clone the repository and enter the repo root:

   ```bash
   git clone https://github.com/<your-org>/KCQRL-RL.git
   cd KCQRL-RL
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the DDPG actor from the configured Google Drive assets:

   ```bash
   python train.py --config configs/default.yaml
   ```

   Training writes the best-reward actor to `outputs/ddpg_actor_best.pt` and the final actor to `outputs/ddpg_actor_final.pt`.

5. Evaluate the trained DDPG actor:

   ```bash
   python evaluate.py --config configs/default.yaml --debug --ddpg-checkpoint outputs/ddpg_actor_best.pt
   ```

If Google Drive assets are unavailable, the scripts print the missing configured paths instead of crashing with import errors. If the DDPG actor checkpoint is unavailable during evaluation, the evaluator prints a warning and falls back to the old heuristic policy.

## Offline evaluation

`evaluation/offline_eval.py` implements a CAT/RL evaluator for these policies:

* `Random`
* `MIRT-MFI`
* `MIRT-KLI`
* `DDPG`
* `OneStepOracle`

For DDPG, evaluation loads the trained LSTM actor checkpoint passed via `--ddpg-checkpoint`, builds candidate vectors from the same representation used during training (`q_matrix + NCDM difficulty + NCDM discrimination`), and selects the nearest candidate item to the actor's ideal action vector.

## Evaluation protocols

KCQRL-RL now keeps two offline evaluation paths:

* **Legacy evaluation** (`python evaluate.py --config configs/default.yaml --debug`) preserves the original CAT/RL replay behavior for backward compatibility. It is useful for sanity checks, but it is **not recommended for paper results** because each policy may be scored on different selected items.
* **`benchmark_v2` paper evaluation** uses a paired, deterministic support/query protocol. For every seed and student, item IDs are first filtered to the shared asset bounds, then a deterministic `student_id + seed` split creates an 80% support/candidate pool and a 20% fixed query set (minimum five query items). All policies receive the same warm-start item, same candidate pool, same query items, and same checkpoints (`0,1,3,5,10,20`). Query responses are hidden from deployable policies and used only for metric computation; `OneStepOracle` is explicitly marked as a privileged-information upper bound.

### Running benchmark_v2

Single-seed debug run:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --protocol benchmark_v2 \
  --policies Random,MIRT-Trace-MFI,MIRT-D-opt,MIRT-MKLI,DDPG \
  --debug \
  --ddpg-checkpoint outputs/ddpg_actor_final.pt \
  --seeds 42 \
  --max-students 20 \
  --steps 0,1,3,5 \
  --output-dir results/benchmark_v2
```

Multi-seed paper run:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --protocol benchmark_v2 \
  --ddpg-checkpoint outputs/ddpg_actor_final.pt \
  --seeds 42,43,44,45,46 \
  --max-students 300 \
  --steps 0,1,3,5,10,20 \
  --output-dir results/benchmark_v2
```

Colab example after mounting Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!python evaluate.py --config configs/default.yaml --protocol benchmark_v2 \
  --ddpg-checkpoint outputs/ddpg_actor_final.pt --seeds 42 --max-students 300
```

`benchmark_v2` writes `aggregate.csv`, `per_seed.csv`, `per_student.csv`, `predictions.csv`, `traces.jsonl`, `run_config.yaml`, `policy_metadata.json`, and `splits_seed*.json` under the configured output directory. Current `MIRT-MFI` and `MIRT-KLI` adapters are metadata-tagged as `implementation: heuristic`; they are simplified proxies, not formal MIRT implementations.


### Benchmark v2 MIRT selector comparison

`benchmark_v2` reports **selector-level** comparisons by default: Random, heuristic selectors, formal MIRT selectors, and DDPG may use different item-selection models, but query-set evaluation is kept on the same frozen NCDM predictor. This isolates the value of the selector while avoiding end-to-end model/evaluator changes. In contrast, an **end-to-end** comparison would evaluate each selector with its own response model (for example, MIRT-selected histories evaluated by MIRT), which answers a different question and is not directly comparable to the DDPG selector-level benchmark.

Formal MIRT selectors (`MIRT-Trace-MFI`, `MIRT-D-opt`, `MIRT-MKLI`) load `assets.mirt_checkpoint`, infer the checkpoint dimensions dynamically, freeze item parameters, and refit an independent test-student theta from support history before every selection. Legacy `MIRT-MFI` and `MIRT-KLI` remain heuristic proxy baselines and are marked as such in policy metadata.
