import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import os

# Opcjonalnie: jeśli YOLOv8 ma być używane
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Detekcja CUDA / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nazwy klas dla klasyfikatora
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'OK', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

# Inicjalizacja GUI
app = tk.Tk()
app.title("Wykrywanie uszkodzeń PCB")
app.geometry("520x540")
app.resizable(False, False)

# Globalne zmienne
image_path = ""
model_path = ""

# GUI Elements
model_label = tk.Label(app, text="Brak modelu")
image_label = tk.Label(app, text="Brak zdjęcia")
image_display = tk.Label(app)
result_label = tk.Label(app, text="Wynik: brak", font=("Arial", 12, "bold"))
fault_detail_label = tk.Label(app, text="Rodzaj uszkodzenia: brak", font=("Arial", 11))

def choose_model():
    global model_path
    path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt *.pth")])
    if path:
        model_path = path
        model_label.config(text=os.path.basename(path))

def choose_image():
    global image_path
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        image_path = path
        image_label.config(text=os.path.basename(path))
        show_image(path)

def show_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((200, 200))
        photo = ImageTk.PhotoImage(img)
        image_display.config(image=photo)
        image_display.image = photo
    except Exception as e:
        messagebox.showerror("Błąd obrazu", str(e))

def run_prediction():
    if not model_path or not image_path:
        messagebox.showwarning("Brak danych", "Wybierz model i zdjęcie.")
        return

    ext = os.path.splitext(model_path)[-1].lower()

    if ext == ".pth":
        run_resnet_prediction()
    elif ext == ".pt" and YOLO_AVAILABLE:
        run_yolo_detection()
    else:
        messagebox.showerror("Nieobsługiwany format", "Nieobsługiwany plik modelu lub brak YOLO.")

def run_resnet_prediction():
    try:
        checkpoint = torch.load(model_path, map_location=device)
        fc_weight = checkpoint.get('fc.weight', None)
        num_classes = fc_weight.size(0) if fc_weight is not None else len(CLASS_NAMES)

        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()

        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        class_name = CLASS_NAMES[prediction] if prediction < len(CLASS_NAMES) else f"Klasa {prediction}"
        if class_name == "OK":
            result_label.config(text="Wynik: Sprawny")
            fault_detail_label.config(text="Rodzaj uszkodzenia: brak")
        else:
            result_label.config(text="Wynik: Uszkodzony")
            fault_detail_label.config(text=f"Rodzaj uszkodzenia: {class_name.replace('_', ' ')}")

    except Exception as e:
        messagebox.showerror("Błąd ResNet", str(e))

def run_yolo_detection():
    try:
        model = YOLO(model_path)
        results = model(image_path, show=True)  # Wyświetlenie obrazu z zaznaczeniami
    except Exception as e:
        messagebox.showerror("Błąd YOLO", str(e))

# GUI layout
tk.Label(app, text="Wybierz plik modelu (.pt lub .pth):").pack(pady=5)
tk.Button(app, text="Wybierz model", command=choose_model).pack()
model_label.pack()

tk.Label(app, text="Wybierz zdjęcie do analizy:").pack(pady=5)
tk.Button(app, text="Wybierz zdjęcie", command=choose_image).pack()
image_label.pack()

image_display.pack(pady=10)

tk.Button(app, text="Wykryj uszkodzenie", command=run_prediction, bg="#4CAF50", fg="white").pack(pady=10)
result_label.pack()
fault_detail_label.pack()

tk.Label(app, text=f"Urządzenie: {device}").pack(pady=10)

app.mainloop()