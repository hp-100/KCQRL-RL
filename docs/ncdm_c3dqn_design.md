# C3DQN-NCDM Design

`C3DQN-NCDM` is **Candidate-Conditioned Cognitive Diagnostic Double DQN**, an NCDM-native adaptive item-selection algorithm. It freezes the NCDM and Q matrix during policy training/evaluation and directly scores the current real candidate set.

## Tensor contract

Let `K = q_matrix.shape[1]`.

* History input: `[B,T,2K+3]`; for K=36 this is `[B,T,75]`.
* Candidate input: `[B,C,2K+1]`; for K=36 this is `[B,C,73]`.
* Global diagnosis input: `[B,2K+1]`; for K=36 this is `[B,73]`.
* Output: RL `q_values` with shape `[B,C]`.

`q_mask` always means the Q-matrix concept pattern. RL `q_values` always means action values. The two names are deliberately never overloaded.

## Cached item features

For each item, features are precomputed once:

```text
q_mask = q_matrix[item_id]
difficulty = sigmoid(ncdm.k_difficulty(item_id))
masked_difficulty = q_mask * difficulty
disc_norm = sigmoid(ncdm.e_discrimination(item_id))
candidate_feature = concat(q_mask, masked_difficulty, disc_norm)
```

The normalized discrimination remains in `[0,1]` for policy inputs.

## History features

Each observed interaction token is:

```text
history_feature = concat(q_mask, masked_difficulty, disc_norm, response, position_fraction)
```

`response` is the raw scalar `0.0/1.0`; there is no response embedding and no semantic feature.

## Candidate-conditioned cross attention

The network projects history and candidate features to dimension `D`, encodes history with a padding-aware Transformer encoder, then computes:

```text
Q = candidate_embeddings        [B,C,D]
K = encoded_history             [B,T,D]
V = encoded_history             [B,T,D]
candidate_context = Attention(Q,K,V)
```

Thus the same student history can yield different context vectors for different candidate items.

## Dueling Double DQN

The value head uses masked history pooling plus global diagnosis embedding:

```text
V(s) = value_head(pool(encoded_history), global_embedding)
```

The advantage head uses candidate embedding, candidate-conditioned context, broadcast global embedding, and explicit cognitive interaction features:

```text
mastered = mastery * candidate_q_mask
weakness = (1 - mastery) * candidate_q_mask
difficulty_gap = mastered - candidate_masked_difficulty
A(s,i) = advantage_head(candidate_embedding, candidate_context, global_embedding, mastered, weakness, difficulty_gap)
q_values = V(s) + A(s,i) - masked_mean_i A(s,i)
```

Padding candidates are assigned finite tiny values (`-1e9`) and are excluded from masked mean, argmax, Double DQN action selection and loss gathering.

Double DQN target uses the online network for next-action selection and the target network for next-action valuation:

```text
next_action = argmax_i online(next_state)_i
target = reward + gamma * (1 - done) * target_net(next_state)[next_action]
```

## Reward

The reward has three components:

```text
prediction_gain = clip(prediction_scale * (query_nll_before - query_nll_after), -1, 1)
diagnosis_gain = mastery_entropy_before - mastery_entropy_after
coverage_gain = new_concepts.sum() / max(1, selected_q_mask.sum())
total = prediction_weight * prediction_gain + diagnosis_weight * diagnosis_gain + coverage_weight * coverage_gain
```

The entropy term is a **diagnosis-confidence proxy** only; it is not true mastery error or true diagnostic accuracy.

Query labels are used only for policy-training reward and validation/final evaluation. Query item IDs, query responses, future candidate responses and query losses are not policy-network inputs.

## NCDM-native track

The `ncdm_native` track requires only `q_matrix`, `ncdm_checkpoint` and test sequences. It does not require a MIRT checkpoint, semantic item bank or semantic embeddings. The first policies are `Random-NCDM` and `C3DQN-NCDM`.

## What is intentionally not used

* semantic embeddings;
* response embeddings;
* continuous actions;
* nearest-neighbor mapping from virtual items to real items.

## Difference from NCAT

NCAT uses item ID embeddings and commonly outputs fixed full-item-bank Q-values. C3DQN-NCDM uses explicit frozen NCDM parameters, shares a scorer across dynamic candidate sets, and makes each candidate query the student's encoded history independently.

## Known limitations

* NCDM alpha fitting may be the main runtime bottleneck.
* Diagnosis entropy is not true diagnostic accuracy.
* Large candidate pools may need deterministic prefiltering later.
* This first version does not include NCAT's correct/incorrect dual-channel contradiction module.

## Efficient Set-C3DQN-NCDM compute protocol

Formal training uses the shared-prefilter Efficient Set-C3DQN-NCDM protocol: a vectorized NCDM diagnostic prefilter first reduces the full real candidate pool, then candidate-history attention and the ISAB candidate-set encoder score only the deterministic filtered pool. Full candidate self-attention is reserved for small ablations and is capped by `full_attention_max_candidates`.

Default smoke and pilot configs use ISAB, one set layer, small inducing-point counts, AMP where CUDA is available, incremental alpha fitting, and disabled large debug tensor retention. Benchmark comparisons between base C3DQN and Set-C3DQN should use either a full-candidate protocol for both methods or the same saved shared-prefilter manifest/protocol for both methods; never compare a full-pool policy to a Top-K-prefiltered policy without labeling the protocol.

The compute profile entry point is `scripts/profile_ncdm_c3dqn_compute.py`; it reports training-step milliseconds, peak memory, parameter count, and candidate count for the required Top-K/ISAB/full-attention cases.
