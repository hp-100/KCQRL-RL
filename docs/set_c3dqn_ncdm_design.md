# Set-C3DQN NCDM Design

Set-C3DQN keeps the base `CandidateConditionedNCDMQNetwork` separate from `SetConditionedNCDMQNetwork`. Candidate set interaction is applied after the candidate has queried encoded student history, so ISAB/full self-attention receives the student-conditioned local candidate representation rather than static candidate features.

Relative cognitive features are optional and contain five candidate-level values: `novelty_ratio`, `covered_overlap_ratio`, `mean_mastery_gap`, `weakness_targeting`, and `concept_count_norm`. When disabled, this projection is absent.

`forward_chunked` preserves the same public contract as `forward`; for set encoders that require all candidates together it delegates to full forward to avoid overstating streaming attention. This keeps dueling advantage centering mathematically global over the full candidate set.
