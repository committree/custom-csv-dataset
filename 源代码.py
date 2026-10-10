import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
x1 = np.random.randn(50, 2) + np.array([2, 2])
x2 = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([x1, x2])
y = np.array([0] * 50 + [1] * 50)

df = pd.DataFrame(X, columns=["feature1", "feature2"])
df["label"] = y
# 自动取当前脚本所在的绝对路径生成CSV，彻底避免相对路径坑
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_full_path = os.path.join(current_dir, "my_custom_data.csv")
df.to_csv(csv_full_path, index=False)
print(f"✅ CSV已自动生成在: {csv_full_path}")


class MyDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor([row["feature1"], row["feature2"]], dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label


dataset = MyDataset(csv_full_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


model = TwoLayerNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch_num = 50
loss_history = []

for epoch in range(epoch_num):
    total_loss = 0.0
    for batch_feat, batch_label in train_loader:
        optimizer.zero_grad()
        pred = model(batch_feat)
        loss = criterion(pred, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epoch_num}], 平均Loss: {avg_loss:.4f}")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("自定义CSV数据集训练Loss变化")
plt.show()
