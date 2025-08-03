
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === Konfiguracja ===
dataset_dir = r"C:\Users\hubert\Downloads\PCB_DATASET\dataset_final"
img_size = 256
batch_size = 8
epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformacje
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# === Dataset
train_data = datasets.ImageFolder(dataset_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# === Prosta CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)

# === Funkcja straty i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# === Trening
print("🔁 Rozpoczynanie treningu...")
start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoka {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

total_time = time.time() - start_time
print(f"⏱ Trening zakończony w {total_time:.1f} sekund.")
torch.save(model.state_dict(), "cnn_model.pth")
print("💾 Model zapisany jako cnn_model.pth")
