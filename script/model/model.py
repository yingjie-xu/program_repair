import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.o2o = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = output[0, :]
        hidden = hidden[0, :]
        cell = cell[0, :]
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
      
    def initCell(self):
        return torch.zeros(1, self.cell_size)

