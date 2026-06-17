# Asset Manifest

This repository stores code only. Data files, checkpoints, and precomputed embeddings are stored in Google Drive and should not be committed to GitHub.

## Google Drive base path

When running in Google Colab, mount Google Drive first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then use the following base path:

```text
/content/drive/MyDrive/Colab_Projects/KCQRL-main/data/XES3G5M
```

## Important external assets

```text
metadata/q_matrix_multihot_36_expert.pt
metadata/item_bank_128d.npy
metadata/item_features_768d.npy
metadata/ncdm_model_36d_expert_best.pt
metadata/mirt_model_36d.pt
kc_level/train_valid_sequences.csv
kc_level/test.csv
```

## Notes

* `q_matrix_multihot_36_expert.pt`: 36-dimensional expert Q-matrix.
* `item_bank_128d.npy`: compressed 128-dimensional question semantic embeddings.
* `item_features_768d.npy`: original 768-dimensional LLM-based item features.
* `ncdm_model_36d_expert_best.pt`: trained NCDM checkpoint.
* `mirt_model_36d.pt`: trained MIRT checkpoint.
* `train_valid_sequences.csv`: training/validation student response sequences.
* `test.csv`: test student response sequences.

These paths should be loaded through `configs/default.yaml`.

Do not upload `.pt`, `.npy`, `.csv`, checkpoint files, embeddings, or raw data to GitHub.
