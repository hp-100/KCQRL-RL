import torch
import torch.nn as nn

class LSTMActor(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, state_dim)

    def forward(self, state_seq):
        out, _ = self.lstm(state_seq)
        out = out[:, -1, :]
        return self.fc(out)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.fc(x)
