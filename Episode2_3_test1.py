import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        return t

network=Network()

for param in network.parameters():
    print(param.shape)

for name, param in network.named_parameters():
    print(name, '\t\t', param.shape)

# print(network.out)
# print(network.out.weight)
# print(network.out.weight.shape)

# print(network.conv2.weight[0][0])
# print(network.conv2.weight[1][0])

# print(network.conv2)
# print(network.fc1)
# print(network.fc2)
# print(network.out)
