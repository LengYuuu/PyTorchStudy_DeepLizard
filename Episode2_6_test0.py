import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)  # call the conv1 layer
        t = F.relu(t)  # activation function
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # max pooling

        # (3) hidden conv layer
        t = self.conv2(t)  # call the conv2 layer
        t = F.relu(t)  # activation function
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # max pooling

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)  # flatten （为啥不用torch.flatten()？）
        t = self.fc1(t)  # linear mapping
        t = F.relu(t)  # activation function

        # (5) hidden linear layer
        t = self.fc2(t)  # linear mapping
        t = F.relu(t)  # activation function

        # (6) output layer
        t = self.out(t)  # linear mapping
        # t = F.softmax(t, dim=1)        # not use here

        return t

network=Network()
