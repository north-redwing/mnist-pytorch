import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, X1):
        Z1 = self.linear1(X1)
        X2 = F.relu(Z1)
        Z2 = self.linear2(X2)
        Y = F.softmax(Z2, dim=-1)
        return Y
