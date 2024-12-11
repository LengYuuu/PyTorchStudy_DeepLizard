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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)    # step 1: Get batch from the training set.
optimizer = optim.Adam(network.parameters(), lr=0.005)


for epoch in range(5):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:  # step 6: Repeat steps 1-5 until one epoch is completed.
        images, labels = batch

        preds = network(images) # step 2: Pass batch to network.
        loss = F.cross_entropy(preds, labels)   # step 3: Calculate the loss(difference between the predicted values and the true values).

        optimizer.zero_grad()
        loss.backward() # step 4: Calculate the gradient of the loss function w.r.t the network's weights.
        optimizer.step()    # step 5: Update the weights using the gradients to reduce the loss.

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(f"epoch: {epoch}, average loss: {total_loss/len(train_loader):.10f}, average correct: {total_correct/len(train_loader):.10f}")


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])

    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds

prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
train_preds = get_all_preds(network, prediction_loader)

print(f"train_set.targets.shape: {train_set.targets.shape}")
print(f"train_set.targets: {train_set.targets}")
print(f"train_preds.argmax(dim=1).shape: {train_preds.argmax(dim=1).shape}")
print(f"train_preds.argmax(dim=1): {train_preds.argmax(dim=1)}")

stacked = torch.stack(
    (
        train_set.targets,
        train_preds.argmax(dim=1)
    ),
    dim=1
)

print(f"stacked.shape: {stacked.shape}")
print(f"stacked: {stacked}")

cmt = torch.zeros(10, 10, dtype=torch.int32)
print(cmt)

for p in stacked:
    tl, pl = p.tolist() # true label & predict label
    cmt[tl, pl] = cmt[tl, pl] + 1

print(cmt)
