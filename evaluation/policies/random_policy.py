from __future__ import annotations
import random
from .base import BaseCATPolicy, PolicyMetadata

class RandomPolicy(BaseCATPolicy):
    name = "Random"
    metadata = PolicyMetadata(name=name)
    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context)
        self.rng = random.Random(f"{student_id}:{seed}:random")
    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        return int(self.rng.choice(list(candidate_item_ids)))


class RandomMIRTPolicy(RandomPolicy):
    name = "Random-MIRT"
    metadata = PolicyMetadata(name=name, implementation="random_mirt", selection_model="mirt", evaluator_model="mirt", uses_query_labels=False)
