import torch
import torch.nn as nn


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(784, 16)
        self.hidden_layer = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 10)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.input_layer(x)
        x = self.sigmoid(x)
        x = self.hidden_layer(x)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        return x
