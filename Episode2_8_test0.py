import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from Episode2_7_test0 import label

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


torch.set_grad_enabled(False)

network = Network()

data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=20
)

batch = next(iter(data_loader))
images, labels = batch
# print(images.shape)
# print(labels.shape)

preds = network(images)

# print(preds.shape)
# print(preds)
# print(labels)
# print(preds.argmax(dim=1))
# print(F.softmax(preds, dim=1))
# print(F.softmax(preds, dim=1).sum())

print(preds.argmax(dim=1))
print(labels)
print(preds.argmax(dim=1).eq(labels))
print(preds.argmax(dim=1).eq(labels).sum())

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

print(get_num_correct(preds, labels))