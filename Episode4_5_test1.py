import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Step 1: 计算原始数据的均值和标准差
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), num_workers=4)
data = next(iter(loader))

# 打印原始数据的均值和标准差
print("原始数据均值: ", data[0].mean())
print("原始数据标准差: ", data[0].std())

# Step 2: 使用计算的均值和标准差进行标准化
train_set_normal = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((data[0].mean(),), (data[0].std(),))  # 应用标准化
    ])
)

# Step 3: 重新计算标准化后的均值和标准差
loader_normal = torch.utils.data.DataLoader(train_set_normal, batch_size=len(train_set_normal), num_workers=4)
data_normal = next(iter(loader_normal))

# 打印标准化后的均值和标准差
print("标准化后的数据均值: ", data_normal[0].mean())
print("标准化后的数据标准差: ", data_normal[0].std())

# 显示标准化前后的数据分布
plt.hist(data[0].flatten(), bins=100, alpha=0.5, label="原始数据")
plt.hist(data_normal[0].flatten(), bins=100, alpha=0.5, label="标准化数据")
plt.legend()
plt.show()