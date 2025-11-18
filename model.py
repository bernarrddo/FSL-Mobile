import torch
import torch.nn as nn

class ModifiedLSTM(nn.Module):
    def __init__(self, input_size=188, hidden_size=256, num_layers=2, num_classes=33):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # add seq dim if needed
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out