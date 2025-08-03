
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# === Konfiguracja
test_dir = r"C:\Users\hubert\Downloads\PCB_DATASET\test"
model_path = "cnn_model.pth"
img_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Klasy
classes = ['Missing_hole', 'Mouse_bite', 'OK', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

# === Transformacje
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# === Model
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
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Testowanie
results = []
correct = 0
incorrect = 0

with torch.no_grad():
    for fname in os.listdir(test_dir):
        if not fname.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(test_dir, fname)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        predicted_label = classes[pred_class_idx]

        # Oczekiwana klasa z nazwy pliku
        expected_label = None
        for label in classes:
            if label.lower() in fname.lower():
                expected_label = label
                break

        is_correct = predicted_label == expected_label
        if is_correct:
            correct += 1
        else:
            incorrect += 1

        results.append({
            "Plik": fname,
            "Przewidziane": predicted_label,
            "Oczekiwane": expected_label,
            "Poprawnie?": is_correct
        })

# === Zapis do Excela
df = pd.DataFrame(results)
df.to_excel("cnn_wyniki_testu.xlsx", index=False)
print("✅ Wyniki zapisane do cnn_wyniki_testu.xlsx")

# === Wykres
labels = ["Poprawnie", "Błędnie"]
values = [correct, incorrect]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Skuteczność klasyfikacji CNN")
plt.axis("equal")
plt.savefig("cnn_skutecznosc.png")
plt.close()
print("📊 Diagram zapisany do cnn_skutecznosc.png")
