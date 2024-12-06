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
    ])
)

how_many_to_plot = 20

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True
)

plt.figure(figsize=(40,25))
for i, batch in enumerate(train_loader, start=1):
    image, label = batch
    plt.subplot(5,5,i)
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title(train_set.classes[label.item()], fontsize=28)
    if i >= how_many_to_plot: break
plt.show()
