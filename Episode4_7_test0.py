import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict


torch.set_printoptions(linewidth=200)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

image, label = train_set[0]
image = image.unsqueeze(0)


# method 1
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


torch.manual_seed(50)
network = Network()

# method 2
torch.manual_seed(50)
sequential1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=12 * 4 * 4, out_features=120),
    nn.ReLU(),
    nn.Linear(in_features=120, out_features=60),
    nn.ReLU(),
    nn.Linear(in_features=60, out_features=10)
)

# method 3
torch.manual_seed(50)
layers = OrderedDict([
    ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)),
    ('relu1', nn.ReLU()),
    ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
    ('conv2', nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)),
    ('relu2', nn.ReLU()),
    ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
    ('flatten', nn.Flatten()),
    ('fc1', nn.Linear(in_features=12 * 4 * 4, out_features=120)),
    ('relu3', nn.ReLU()),
    ('fc2', nn.Linear(in_features=120, out_features=60)),
    ('relu4', nn.ReLU()),
    ('out', nn.Linear(in_features=60, out_features=10))
])
sequential2 = nn.Sequential(layers)

# method 4
torch.manual_seed(50)
sequential3 = nn.Sequential()
sequential3.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5))
sequential3.add_module('relu1', nn.ReLU())
sequential3.add_module('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
sequential3.add_module('conv2', nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5))
sequential3.add_module('relu2', nn.ReLU())
sequential3.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
sequential3.add_module('flatten', nn.Flatten())
sequential3.add_module('fc1', nn.Linear(in_features=12 * 4 * 4, out_features=120))
sequential3.add_module('relu3', nn.ReLU())
sequential3.add_module('fc2', nn.Linear(in_features=120, out_features=60))
sequential3.add_module('relu4', nn.ReLU())
sequential3.add_module('out', nn.Linear(in_features=60, out_features=10))

# test
print(network)
print(sequential1)
print(sequential2)
print(sequential3)

print(network(image))
print(sequential1(image))
print(sequential2(image))
print(sequential3(image))
