import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=1, batch_first=True)
        self.o2o = nn.Linear(hidden_size * 2 + cell_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output, hidden_squeezed, cell_squeezed = \
            output.squeeze(1), hidden.squeeze(0), cell.squeeze(0)
        output_combined = torch.cat(
            (output, hidden_squeezed, cell_squeezed), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.cell_size)
