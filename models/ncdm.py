import torch
import torch.nn as nn

class OfficialNCDM(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n):
        super().__init__()

        self.knowledge_dim = knowledge_n

        self.student_emb = nn.Embedding(student_n, knowledge_n)
        self.k_difficulty = nn.Embedding(exer_n, knowledge_n)
        self.e_discrimination = nn.Embedding(exer_n, 1)

        self.fc1 = nn.Linear(knowledge_n, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, stu_id, exer_id, kn_emb):

        stu = torch.sigmoid(self.student_emb(stu_id))
        diff = torch.sigmoid(self.k_difficulty(exer_id))
        discr = torch.sigmoid(self.e_discrimination(exer_id)) * 10

        x = discr * (stu - diff) * kn_emb

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        return torch.sigmoid(self.fc3(x)).squeeze(-1)

