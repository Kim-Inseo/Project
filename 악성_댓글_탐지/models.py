import torch
from torch.nn import LSTM, Module, Linear

class CustomModel(Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, device, num_layers, bidirectional):
        super(CustomModel, self).__init__()
        self.device = device

        self.mul = (2 if bidirectional else 1)

        self.lstm = LSTM(input_size=embed_dim,
                         hidden_size=hidden_dim,
                         num_layers=num_layers,
                         bidirectional=bidirectional,
                         batch_first=True)
        self.fc = Linear(hidden_dim*self.mul, output_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        hidden_0 = torch.zeros(self.num_layers*self.mul, x.size(0), self.hidden_dim).to(self.device)
        cell_0 = torch.zeros(self.num_layers*self.mul, x.size(0), self.hidden_dim).to(self.device)
        # (num layers * bidirectional, batch size, hidden dim)

        out_lstm, _ = self.lstm(x, (hidden_0, cell_0))

        output = self.fc(out_lstm[:, -1, :])

        return output