from evaluation.policies.random_policy import RandomPolicy

def test_normal_policy_context_has_no_query_responses():
    p=RandomPolicy(); p.reset('s',42,{})
    ctx={'query_item_ids':[1,2]}
    assert 'query_responses' not in ctx
    assert p.select([3,4], [], [], ctx) in [3,4]
