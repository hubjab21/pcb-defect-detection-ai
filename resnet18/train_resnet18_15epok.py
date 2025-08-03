
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # === Konfiguracja ===
    dataset_dir = r"C:\Users\hubert\Downloads\ai\PCB_DATASET\dataset_final"
    img_size = 256
    batch_size = 64  # zwiększony batch size
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Używane urządzenie:", device)

    # === Transformacje
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # === Dataset
    train_data = datasets.ImageFolder(dataset_dir, transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True  # optymalizacja CPU → GPU
    )

    # === Model: ResNet18 z 7 klasami
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)

    # === Funkcja straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # === Trening
    print("🔁 Rozpoczynanie treningu...")
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start = time.time()
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        print(f"Epoka {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, ⏱ {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"⏱ Trening zakończony w {total_time:.1f} sekund.")
    torch.save(model.state_dict(), "resnet18_model_opt.pth")
    print("💾 Model zapisany jako resnet18_model_opt.pth")

if __name__ == "__main__":
    main()
