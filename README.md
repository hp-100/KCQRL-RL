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
