import time

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


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


torch.set_printoptions(linewidth=120)

train_set_not_normal = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
        # normalize

    ])
)

# Easy way: Calculate the mean and standard deviation using the torch method
loader = torch.utils.data.DataLoader(train_set_not_normal, batch_size=len(train_set_not_normal), num_workers=4)
data = next(iter(loader))

train_set_normal = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor(),
        # normalize
        transforms.Normalize((data[0].mean()), (data[0].std()))
    ])
)

# Easy way: Calculate the mean and standard deviation using the torch method
loader_normal = torch.utils.data.DataLoader(train_set_normal, batch_size=len(train_set_normal), num_workers=4)
data_normal = next(iter(loader_normal))


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


params = OrderedDict(
    train_set=[train_set_not_normal, train_set_normal],
    lr=[.01,.005],
    batch_size=[100,200],
    shuffle=[True],
    epoch=[200],
    num_workers=[4],
    device=["cuda"]
)


def main():
    # for run in RunBuilder.get_runs(params):
    #     comment = f'-{run}'
    #     print(comment)

    results = []

    for run in RunBuilder.get_runs(params):
        comment = f'-{run}'
        print(f"\n")
        print(comment)

        network = Network()

        reading_start_time = time.time()
        train_loader = torch.utils.data.DataLoader(dataset=run.train_set,
                                                   batch_size=run.batch_size,
                                                   shuffle=run.shuffle,
                                                   num_workers=run.num_workers)  # step 1: Get batch from the training set.
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        device = run.device
        network.to(device)
        reading_end_time = time.time()
        # print(f"reading train loader time: {reading_end_time - reading_start_time:.10f}")

        train_start_time = time.time()
        for epoch in range(run.epoch):  # step 7: Repeat steps 1-6 for as many epochs required to obtain the desired level of accuracy.
            total_loss = 0
            total_correct = 0

            for batch in train_loader:  # step 6: Repeat steps 1-5 until one epoch is completed.
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                preds = network(images)  # step 2: Pass batch to network.
                loss = F.cross_entropy(preds, labels)  # step 3: Calculate the loss(difference between the predicted values and the true values).

                optimizer.zero_grad()
                loss.backward()  # step 4: Calculate the gradient of the loss function w.r.t the network's weights.
                optimizer.step()  # step 5: Update the weights using the gradients to reduce the loss.

                total_loss += loss.item()
                total_correct += get_num_correct(preds, labels)

            print(f"epoch: {epoch:3d}, average loss: {total_loss / len(train_loader):.10f}, average correct: {total_correct / len(train_loader):.10f}")

            results.append({
                'epoch': epoch,
                'lr': run.lr,
                'batch_size': run.batch_size,
                'average_loss': (total_loss / len(train_loader)),
                'accuracy': (total_correct / len(train_loader) / run.batch_size),
                'train_set': 'not normal' if run.train_set == train_set_not_normal else 'normal'
            })

        train_end_time = time.time()
        print(f"train time: {train_end_time - train_start_time:.10f}")

    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    print("\nSorted Results (by average correct):")
    for result in sorted_results:
        print(f"epoch: {result['epoch']:3d}, learning rate: {result['lr']:.3f}, batch size: {result['batch_size']}, average loss: {result['average_loss']:.10f}, accuracy: {result['accuracy']:.4f}, train_set: {result['train_set']}")


if __name__ == '__main__':
    main()
