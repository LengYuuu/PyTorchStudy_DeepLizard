import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # short for optimizer. This can give us access to the optimizer we will use to update weights.

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
from collections import namedtuple
from itertools import product


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.reshape(-1, 12 * 4 * 4)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


network = Network()


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

params = OrderedDict(lr = [.01,.005], batch_size = [100,500], device = ["cuda","cpu"], epoch = [100,200])

runs = RunBuilder.get_runs(params)
for run in RunBuilder.get_runs(params):
    comment = f'-{run}'
    print(comment)
