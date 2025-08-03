
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
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

# === Model: EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
model = model.to(device)

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
torch.save(model.state_dict(), "efficientnet_b0_model.pth")
print("💾 Model zapisany jako efficientnet_b0_model.pth")
