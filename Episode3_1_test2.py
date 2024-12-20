import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # short for optimizer. This can give us access to the optimizer we will use to update weights.

import torchvision
import torchvision.transforms as transforms

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

# print(network.conv1.weight.requires_grad)  # 应该是 True

data_loader = torch.utils.data.DataLoader(train_set, batch_size=100)    # step 1: Get batch from the training set.
batch = next(iter(data_loader))
images, labels = batch

preds = network(images) # step 2: Pass batch to network.
loss = F.cross_entropy(preds, labels)   # step 3: Calculate the loss(difference between the predicted values and the true values).
print(loss.item())

# print(network.conv1.weight.grad)
loss.backward() # step 4: Calculate the gradient of the loss function w.r.t the network's weights.
# print(network.conv1.weight.grad)
# print(network.conv1.weight.grad.shape)
# print(network.conv1.weight.shape)

optimizer = optim.Adam(network.parameters(), lr=0.01)
print(loss.item())
print(get_num_correct(preds, labels))
optimizer.step()    # step 5: Update the weights using the gradients to reduce the loss.
preds = network(images)
loss = F.cross_entropy(preds, labels)
print(loss.item())
print(get_num_correct(preds, labels))
