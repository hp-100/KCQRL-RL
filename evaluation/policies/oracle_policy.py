from __future__ import annotations
from .base import BaseCATPolicy, PolicyMetadata

class OneStepOraclePolicy(BaseCATPolicy):
    name = "OneStepOracle"
    metadata = PolicyMetadata(name=name, implementation="one_step_query_nll_upper_bound", uses_privileged_information=True, uses_query_labels=True, notes="Uses logged candidate responses and query labels for one-step lookahead upper bound.")
    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        response_lookup = context["candidate_response_lookup"]
        score_after = context["query_nll_after_history"]
        best_item, best_nll = None, float("inf")
        for item in candidate_item_ids:
            nll = score_after(list(history_item_ids) + [int(item)], list(history_responses) + [float(response_lookup[int(item)])])
            if nll < best_nll:
                best_item, best_nll = int(item), nll
        return int(best_item)
