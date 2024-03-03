import torch.nn as nn

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate, device):  # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(device)
        self.dropout1 = nn.Dropout(p=dropout_rate).to(device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(device)
        self.dropout2 = nn.Dropout(p=dropout_rate).to(device)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(
            self.conv1(inputs.transpose(-1, -2))))))
        # as Conv1D requires (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SimpleFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate, device):
        super(SimpleFeedForward, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units).to(device),
            nn.ReLU(),
            nn.Dropout(dropout_rate).to(device),
            nn.Linear(hidden_units, hidden_units).to(device),
            nn.ReLU(),
            nn.Dropout(dropout_rate).to(device)
        )

    def forward(self, inputs):
        outputs = self.ffn(inputs)
        outputs += inputs
        return outputs