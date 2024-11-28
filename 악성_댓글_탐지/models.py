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
        '''
        :param x: 3차원 Tensor
                  shape: (batch 크기, sequence 길이, 임베딩 벡터 크기)
        :return: 2차원 Tensor
                 shape: (batch 크기, 2)
        '''
        hidden_0 = torch.zeros(self.num_layers*self.mul, x.size(0), self.hidden_dim).to(self.device)
        cell_0 = torch.zeros(self.num_layers*self.mul, x.size(0), self.hidden_dim).to(self.device)
        # 각각의 shape: (layers 개수 * bidirectional, batch 크기, hidden dim)

        out_lstm, _ = self.lstm(x, (hidden_0, cell_0))
        # shape: (batch 크기, sequence 길이, hidden dim * bidirectional)

        output = self.fc(out_lstm[:, -1, :])
        # 마지막 시점의 결과 값(shape: (batch 크기, hidden dim * bidirectional))만 사용하여 FC layer에 넣는다

        return output