import torch.nn as nn
from utils.metric import seed_everything

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate, device): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        # seed_everything(42)
        self.device = device
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(device)
        self.dropout1 = nn.Dropout(p=dropout_rate).to(device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(device)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))).to(self.device)
        outputs = outputs.transpose(-1, -2).to(self.device) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs