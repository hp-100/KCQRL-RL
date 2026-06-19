from evaluation.policies.oracle_policy import OneStepOraclePolicy

def test_oracle_minimizes_nll_not_max_error():
    p=OneStepOraclePolicy()
    ctx={
      'candidate_response_lookup': {1:0.0, 2:1.0},
      'query_nll_after_history': lambda hi,hr: {1:0.1, 2:2.0}[hi[-1]],
    }
    assert p.select([1,2], [], [], ctx) == 1
    assert p.metadata.uses_privileged_information is True
