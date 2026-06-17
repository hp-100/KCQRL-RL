# KCQRL-RL Project Specification

## Goal

Build a research-grade reinforcement-learning-based computerized adaptive testing framework for knowledge tracing and adaptive question selection.

The system should select the next question for a student based on historical responses, cognitive diagnosis states, question embeddings, and knowledge concepts.

## Core components

### 1. Cognitive diagnosis model

Use NCDM as the main student modeling component.

Expected inputs:

* student history
* item IDs
* Q-matrix
* item difficulty
* item discrimination

Expected output:

* predicted probability of correctness
* fitted student ability vector alpha

### 2. Baselines

Implement and evaluate:

* Random
* MIRT-MFI
* MIRT-KLI
* DDPG / RL policy
* One-step oracle

### 3. Question representation

The project uses precomputed question embeddings:

* original LLM-based item features: 768-dimensional
* compressed item bank: 128-dimensional

The 128-dimensional item bank should be loaded from the external asset path and used by the policy/state builder.

### 4. Reinforcement learning policy

The RL policy should select questions from a candidate item pool.

The old DDPG implementation may be used as reference, but the new system should be modular:

* environment
* reward
* agent
* state builder
* evaluation

### 5. Reward

The main reward should be aligned with downstream prediction improvement:

```text
reward = BCE_before_on_heldout - BCE_after_on_heldout
```

An optional small coverage bonus may be added:

```text
reward = nll_gain + lambda_cov * normalized_new_kc
```

Entropy gain should only be used as a diagnostic or ablation reward.

### 6. Evaluation protocol

All methods must share the same evaluation episode:

* same student
* same initial item
* same candidate pool
* same heldout set
* same test length

Metrics:

* AUC
* Accuracy
* NLL / BCE
* Brier score
* Balanced accuracy
* paired bootstrap confidence interval by student

### 7. Colab execution

The system should run in Google Colab after mounting Google Drive.

All external paths must be configured in:

```text
configs/default.yaml
```

The repository should not contain raw data, checkpoints, or embedding files.
