import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import numpy as np

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

sample = next(iter(train_set))
print('len:', len(sample))

image, label = sample
print('types:', type(image), type(label))
print('shape:', image.shape)

print('label:', label)

plt.imshow(image.squeeze(), cmap="gray")
plt.show()
