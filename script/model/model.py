import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=1, batch_first=True).to(device)
        self.o2o = nn.Linear(hidden_size * 2 + cell_size,
                             output_size).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.softmax = nn.LogSoftmax(dim=1).to(device)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output, hidden_squeezed, cell_squeezed = \
            output.squeeze(1).to(self.device), hidden.squeeze(
                0).to(self.device), cell.squeeze(0).to(self.device)
        output_combined = torch.cat(
            (output, hidden_squeezed, cell_squeezed), 1).to(self.device)
        output = self.o2o(output_combined).to(self.device)
        output = self.dropout(output).to(self.device)
        output = self.softmax(output).to(self.device)
        return output, hidden, cell

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(self.device)

    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.cell_size).to(self.device)
