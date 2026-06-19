from __future__ import annotations
import numpy as np
from .base import BaseCATPolicy, PolicyMetadata

class HeuristicMIRTPolicy(BaseCATPolicy):
    def __init__(self, name: str):
        self.name = name
        self.metadata = PolicyMetadata(name=name, implementation="heuristic", notes="Simplified proxy, not a formal MIRT implementation.")
    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        preds = context["predict_history"](history_item_ids, history_responses, candidate_item_ids)
        if self.name == "MIRT-MFI":
            idx = int(np.argmax([p * (1 - p) for p in preds]))
        else:
            idx = int(np.argmax([abs(p - 0.5) for p in preds]))
        return int(list(candidate_item_ids)[idx])
