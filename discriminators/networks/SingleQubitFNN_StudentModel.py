import torch.nn as nn

class SingleQubitStudentModel(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_s_1 = 64
        hidden_s_2 = 32

        hidden_size = [hidden_s_1, hidden_s_2]
        print("Hidden Layer Size:", hidden_s_1, hidden_s_2)

        super(SingleQubitStudentModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.l2(x)))
        x = self.dropout(x)
        x = self.l3(x)
        return x
