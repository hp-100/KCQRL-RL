from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence

@dataclass
class PolicyMetadata:
    name: str
    implementation: str = "reference"
    uses_privileged_information: bool = False
    notes: str = ""

class BaseCATPolicy:
    name = "Base"
    metadata = PolicyMetadata(name="Base")
    def reset(self, student_id, seed, context):
        self.student_id = student_id
        self.seed = seed
    def select(self, candidate_item_ids: Sequence[int], history_item_ids: Sequence[int], history_responses: Sequence[float], context: dict[str, Any]) -> int:
        raise NotImplementedError
