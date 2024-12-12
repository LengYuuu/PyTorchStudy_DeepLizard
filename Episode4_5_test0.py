import torch

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
        # normalize

    ])
)

# Easy way: Calculate the mean and standard deviation using the torch method
loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), num_workers=4)
data = next(iter(loader))
print(data[0].mean(), data[0].std())

plt.hist(data[0].flatten())
plt.axvline(data[0].mean())
plt.show()


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
loader_normal = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), num_workers=4)
data_normal = next(iter(loader_normal))
print(data_normal[0].mean(), data_normal[0].std())

plt.hist(data_normal[0].flatten())
plt.axvline(data_normal[0].mean())
plt.show()