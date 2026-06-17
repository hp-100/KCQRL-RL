class KTEnv:
    def __init__(self, dataset, ncdm_model):
        self.dataset = dataset
        self.model = ncdm_model

    def reset(self, student_id):
        self.history_q = []
        self.history_r = []
        return self._get_state()

    def step(self, item_id, response):

        self.history_q.append(item_id)
        self.history_r.append(response)

        reward = self._compute_reward()

        return self._get_state(), reward, False, {}

    def _get_state(self):
        return {
            "history_q": self.history_q,
            "history_r": self.history_r
        }
