import torch
from typing import Any
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


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = BasicConv2d(1, 8, kernel_size=3)
        self.hidden_layer_1 = BasicConv2d(8, 16, kernel_size=3)
        self.hidden_layer_2 = BasicConv2d(16, 16, kernel_size=3, stride=2)
        self.hidden_layer_3 = BasicConv2d(16, 16, kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(16, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        # print(x.shape)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.output_layer(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    model = MyNN()
    print("MyNN amount of parameters: ", sum(p.numel() for p in model.parameters()))
    model = MyCNN()
    print("MyCNN amount of parameters: ", sum(p.numel() for p in model.parameters()))
