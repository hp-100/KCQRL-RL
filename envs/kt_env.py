"""Online-style knowledge tracing environment wrapper."""


class KTEnv:
    def __init__(self, dataset=None, ncdm_model=None):
        self.dataset = dataset
        self.model = ncdm_model
        self.history_q = []
        self.history_r = []

    def reset(self, student_id=None):
        self.student_id = student_id
        self.history_q = []
        self.history_r = []
        return self.get_state()

    def step(self, item_id, response):
        self.history_q.append(item_id)
        self.history_r.append(float(response))
        reward = self._compute_reward(response)
        return self.get_state(), reward, False, {"item_id": item_id, "response": response}

    def get_state(self):
        return {"history_q": list(self.history_q), "history_r": list(self.history_r)}

    def _compute_reward(self, response):
        return float(response)
