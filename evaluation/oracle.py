"""Oracle item-selection helpers for offline CAT evaluation."""


def one_step_oracle(env, candidates, predictor=None):
    """Select the candidate with the largest immediate absolute prediction error."""
    if not candidates:
        return None
    if predictor is None:
        return candidates[0]
    return max(candidates, key=lambda item: abs(env.responses.get(item, 0.0) - predictor(item)))
