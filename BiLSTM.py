import torch
import torch.nn as nn
import torch.nn.functional as F



class BiLSTM_Net(torch.nn.Module):
    def __init__(self, in_feature, size_hidden, n_output, n_layer):
        super(BiLSTM_Net, self).__init__()
        self.n_layer = n_layer
        self.size_hidden = size_hidden

        self.lstm = nn.LSTM(input_size=in_feature,
                            hidden_size=size_hidden,
                            num_layers=n_layer,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(size_hidden * 2, n_output)

    def forward(self, x):

        x, _ = self.lstm(x)
        x = F.sigmoid(self.linear(x))

        return x