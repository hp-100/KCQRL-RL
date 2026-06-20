# Efficient Set-C3DQN-NCDM design

Base **C3DQN-NCDM** is already a discrete-action model: the action is the argmax over the currently valid candidate item IDs, and there is no continuous action, nearest-neighbor remapping, fixed item-output layer, item-ID embedding, or semantic embedding. The original `CandidateConditionedNCDMQNetwork` remains the Base architecture and keeps the checkpoint architecture id `candidate_conditioned_attention_dueling_double_dqn`.

**Set-C3DQN-NCDM** adds candidate-set conditional modeling in the independent `SetConditionedNCDMQNetwork` with architecture id `set_conditioned_candidate_attention_dueling_double_dqn`. Base and Set checkpoints are intentionally incompatible and loaders dispatch strictly by architecture id.

## Candidate set encoder modes

* `none`: skips candidate-candidate interaction and is used for ablations.
* `full_self_attention`: exact self-attention for small pools only; the model rejects pools larger than `full_attention_max_candidates` because complexity is approximately `O(C²D)`.
* `isab`: default efficient set encoder. ISAB uses learned inducing points with two MABs, residual connections, LayerNorm, FFN, and dropout. Its candidate-set complexity is approximately `O(CMD)`.

## Local candidate representation

The Set encoder input is a local candidate representation, not only candidate-history context. It combines candidate embedding, candidate-history cross-attention context, global diagnosis embedding, and projected cognitive features. Cognitive features concatenate `mastered`, `weakness`, `difficulty_gap`, and optional relative features.

Relative features are: `novelty_ratio`, `covered_overlap_ratio`, `mean_mastery_gap`, `weakness_targeting`, and `concept_count_norm`. They are masked by the item Q-vector and clamp denominators to avoid NaN for all-zero Q-masks.

## Dueling and chunking

Set-aware value heads can include a masked mean pool of set-aware candidates. Advantages are centered by the global masked mean over the full candidate set. `forward_chunked` is mathematically equivalent to full forward in eval mode because it computes the value once, concatenates all raw advantage chunks, and subtracts one global masked advantage mean. Chunking only splits candidate scoring after the Top-K candidate set has been materialized; it does not claim full streaming candidate-history attention. Top-K prefiltering is the primary compute and memory reduction mechanism.

## Protocol

Training and evaluation share `NCDMCandidatePrefilter`: full candidate pool → alpha/mastery/coverage-count scoring → Top-K candidates → network argmax over real candidate IDs. Evaluation reads `candidate_pool_config` and `alpha_fit` from checkpoint metadata. Alpha fitting uses `initial_steps` for first/inconsistent histories and `incremental_steps` with cached alpha when the episode history grows by one.

The model is permutation equivariant over candidates: permuting candidate order permutes Q-values, and padding candidates are masked so they do not affect ISAB summaries or selected item IDs.

## Paper ablations

The supported ablations are Base C3DQN, Set-none, Relative-only, ISAB-only, and Full Efficient Set-C3DQN. All share the same NCDM, Q-matrix, reward, Double-DQN protocol, support/query split, warm start, seed, and Top-K prefilter configuration.
