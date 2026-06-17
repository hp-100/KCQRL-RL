def one_step_oracle(env, candidates):

    best_item = None
    best_gain = -1e9

    base_state = env.get_state()

    for item in candidates:

        gain = simulate_gain(env, item)

        if gain > best_gain:
            best_gain = gain
            best_item = item

    return best_item
