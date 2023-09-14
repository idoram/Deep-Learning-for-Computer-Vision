import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from net import LeNet5
from tqdm import tqdm
import os

# 定义超参数
in_channels = 1
num_classes = 10
batch_size = 6000
num_epochs = 33
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据集
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

# 定义数据集加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义评估函数
def eval(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += images.size(0)
            correct += (predicted == labels).sum().item()   # 用item转化成标量值
    accuracy = correct / total
    return accuracy * 100


model = LeNet5(in_channels, num_classes, "relu").to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(params=model.parameters(), lr=12e-3)

best_accuracy = 0.0                                         # 用于保存最佳准确度
best_epoch = 0                                              # 用于保存最佳准确度对应的epoch


for epoch in range(num_epochs):
    if epoch == 0:
        print('[INFO] Start Train.')
    pbar = tqdm(total=len(train_loader.dataset), desc=f"[INFO] Epoch {epoch + 1}/{num_epochs}", postfix=dict,
                mininterval=0.3)
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        train_loss += loss.item() * images.size(0)

        # 更新进度条
        pbar.set_postfix(**{"Loss": f"{loss.item():.4f}", "Accuracy": f"{eval(model, test_loader, device):.4f}"})
        pbar.update(images.size(0))

    pbar.close()

    train_loss /= len(train_loader.dataset)
    test_acc = eval(model, test_loader, device)

    # print(f"\n[INFO] Epoch: {epoch + 1} / {num_epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f} %")

    # 保存权重
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_epoch = epoch + 1
        if not os.path.exists("./model"):
            os.makedirs("./model")
        torch.save(model.state_dict(), "./model/best_model.pth")

torch.save(model.state_dict(), "./model/last_epoch_model.pth")
print(f"\n[INFO] Best Accuracy achieved at Epoch {best_epoch}, Accuracy: {best_accuracy:.4f} %")

