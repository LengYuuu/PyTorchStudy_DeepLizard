import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # short for optimizer. This can give us access to the optimizer we will use to update weights.

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
optimizer = optim.Adam(network.parameters(), lr=0.01)


for epoch in range(100):
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

stacked = torch.stack(
    (
        train_set.targets,
        train_preds.argmax(dim=1)
    ),
    dim=1
)

cmt = torch.zeros(10, 10, dtype=torch.int32)
print(cmt)

for p in stacked:
    tl, pl = p.tolist() # true label & predict label
    cmt[tl, pl] = cmt[tl, pl] + 1

print(cmt)

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


draw_confusion_matrix(label_true=train_set.targets,
                      label_pred=train_preds.argmax(dim=1),
                      label_name=["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle"],
                      title="Confusion Matrix on FashionMNIST, range = 100, lr = 0.005",
                      pdf_save_path="D:\\Programme\\DeepLearning\\PyTorch\\test1\\Confusion_Matrix_on_FashionMNIST.jpg",
                      dpi=300)
