"""Simple offline knowledge-tracing environment for replayed CAT sequences."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class OfflineKTEnv:
    responses: Dict[int, float]
    history: List[Tuple[int, float]] = field(default_factory=list)

    def reset(self):
        self.history.clear()
        return self.get_state()

    def step(self, item_id: int):
        if item_id not in self.responses:
            raise KeyError(f"No logged response available for item {item_id}")
        response = float(self.responses[item_id])
        self.history.append((item_id, response))
        return self.get_state(), response, False, {"item_id": item_id, "response": response}

    def get_state(self):
        return {"history": list(self.history)}
