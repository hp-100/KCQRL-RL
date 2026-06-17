# KCQRL-RL

A minimal, runnable Knowledge-Concept Question Recommendation Learning (KCQRL-RL) framework for offline computerized adaptive testing (CAT) and reinforcement-learning policy evaluation.

## Repository layout

The project is intended to run from the repository root. Core folders live directly under the root:

```text
agents/ configs/ core/ datasets/ docs/ envs/ evaluation/ legacy/ models/ reward/ scripts/ utils/
```

Large data assets and checkpoints are **not** committed. See `docs/ASSET_MANIFEST.md` and configure all external paths in `configs/default.yaml`.

## Colab quick start

```bash
git clone https://github.com/<your-org>/KCQRL-RL.git
cd KCQRL-RL
pip install -r requirements.txt
```

Mount Google Drive in a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run debug evaluation:

```bash
python scripts/evaluate.py --config configs/default.yaml --debug
```

Run the training scaffold:

```bash
python scripts/train.py --config configs/default.yaml
```

If Google Drive assets are unavailable, the scripts print the missing configured paths instead of crashing with import errors.

## Offline evaluation

`evaluation/offline_eval.py` implements a CAT/RL evaluator for these policies:

* `Random`
* `MIRT-MFI`
* `MIRT-KLI`
* `DDPG`
* `OneStepOracle`

This path does not require tokenizers, transformer models, `json_file_dataset`, `kc_questions_map`, or cluster JSON files.
