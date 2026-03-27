import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import os

# Optional: if YOLOv8 is required
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# CUDA / CPU Detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Names for the Classifier
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'OK', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

# GUI Initialization
app = tk.Tk()
app.title("PCB Inspection using Deep Learning")
app.geometry("620x640")
app.resizable(False, False)

# Global Variables
image_path = ""
model_path = ""

# Theme state
dark_mode = False

# GUI Elements
model_label = tk.Label(app, text="No model loaded")
image_label = tk.Label(app, text="No image selected")
image_display = tk.Label(app)
result_label = tk.Label(app, text="Result: none", font=("Arial", 12, "bold"))
fault_detail_label = tk.Label(app, text="Defect type: none", font=("Arial", 11))

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
        messagebox.showerror("Image Error", str(e))

def run_prediction():
    if not model_path or not image_path:
        messagebox.showwarning("No Data", "Please select a model and an image.")
        return

    ext = os.path.splitext(model_path)[-1].lower()

    if ext == ".pth":
        run_resnet_prediction()
    elif ext == ".pt" and YOLO_AVAILABLE:
        run_yolo_detection()
    else:
        messagebox.showerror("Unsupported Format", "Unsupported model file or YOLO support is missing.")

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
            result_label.config(text="Result: OK")
            fault_detail_label.config(text="Defect type: none")
        else:
            result_label.config(text="Result: Defective")
            fault_detail_label.config(text=f"Defect type: {class_name.replace('_', ' ')}")

    except Exception as e:
        messagebox.showerror("ResNet Error", str(e))

def run_yolo_detection():
    try:
        model = YOLO(model_path)
        results = model(image_path, show=True)  # Display Image with Annotations
    except Exception as e:
        messagebox.showerror("YOLO Error", str(e))

def toggle_theme():
    global dark_mode
    dark_mode = not dark_mode

    if dark_mode:
        bg_color = "#2b2b2b"
        fg_color = "white"
        btn_color = "#444444"
    else:
        bg_color = "#f0f0f0"
        fg_color = "black"
        btn_color = "#e0e0e0"

    app.config(bg=bg_color)

    for widget in app.winfo_children():
        try:
            widget.config(bg=bg_color, fg=fg_color)
        except:
            pass

    theme_button.config(
        text="Light Mode" if dark_mode else "Dark Mode",
        bg=btn_color,
        fg=fg_color
    )

# GUI layout
tk.Label(app, text="Select model file (.pt or .pth):").pack(pady=5)
tk.Button(app, text="Select Model", command=choose_model).pack()
model_label.pack()

tk.Label(app, text="Select image for analysis:").pack(pady=5)
tk.Button(app, text="Select Image", command=choose_image).pack()
image_label.pack()

image_display.pack(pady=10)

tk.Button(app, text="Detect Defect", command=run_prediction, bg="#4CAF50", fg="white").pack(pady=10)
result_label.pack()
fault_detail_label.pack()

tk.Label(app, text=f"Device: {device}").pack(pady=10)

theme_button = tk.Button(app, text="Dark Mode", command=toggle_theme)
theme_button.pack(pady=5)

app.mainloop()
