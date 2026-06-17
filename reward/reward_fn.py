import torch

def nll_gain(model, before_state, after_state, item, response):

    with torch.no_grad():
        before = model.predict(before_state)
        after = model.predict(after_state)

    return before - after


def coverage_bonus(new_items, total_items):
    return len(set(new_items)) / total_items


def reward_fn(nll_gain, coverage, lambda_cov=0.03):
    return nll_gain + lambda_cov * coverage
