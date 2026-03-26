# 🧠 PCB Defect Detection using Deep Learning

This project focuses on detecting defects and incorrect connections in PCB (Printed Circuit Boards) using deep learning techniques.

---

## 📌 Project Goal

The main objective of this project was to design and evaluate multiple deep learning approaches for PCB defect detection and to perform a **comparative analysis of different neural network architectures**.

The study includes both:

* image classification models
* object detection approach (YOLOv8)

---

## 🔬 Research Approach

This project is not only an implementation but also a **technology review and comparison of modern deep learning architectures** applied to PCB inspection.

Different models were trained and evaluated under the same conditions to analyze:

* accuracy
* generalization capability
* suitability for industrial applications

---

## 📊 Dataset

* ~1400 PCB images
* 7 defect classes:

  * Missing_hole
  * Mouse_bite
  * OK
  * Open_circuit
  * Short
  * Spur
  * Spurious_copper

Images were resized to:

* 256x256 (classification)
* 640x640 (YOLOv8)

⚠️ Important limitation:
Dataset contained only ~12 unique PCB boards, which significantly affected generalization. 

---

## 🧠 Models Overview

### 🔹 MLP (256 neurons)

A simple fully connected neural network used as a baseline model.

* no spatial feature extraction
* very limited performance
* accuracy: **10%**

---

### 🔹 Custom CNN

Basic convolutional neural network designed for feature extraction.

* detects edges and local patterns
* lightweight architecture
* accuracy: **30%**

---

### 🔹 ResNet18

Residual neural network using skip connections.

* solves vanishing gradient problem
* allows deeper learning
* accuracy: **30%**

---

### 🔹 EfficientNet-B0 / B1

Modern architecture optimized for performance vs efficiency.

* compound scaling (depth, width, resolution)
* efficient but sensitive to dataset quality
* accuracy:

  * B0: **10%**
  * B1: **20%**

---

### 🔹 MobileNetV2

Lightweight model designed for embedded systems.

* optimized for low computational cost
* suitable for real-time applications
* accuracy: **20%**

---

### 🔹 DenseNet121

Architecture with dense connections between layers.

* strong feature reuse
* improved gradient flow
* accuracy: **20%**

---

## 📉 Classification Results Summary

| Model             | Accuracy |
| ----------------- | -------- |
| ResNet18          | 30%      |
| CNN (custom)      | 30%      |
| MobileNetV2       | 20%      |
| DenseNet121       | 20%      |
| EfficientNet-B1   | 20%      |
| EfficientNet-B0   | 10%      |
| MLP (256 neurons) | 10%      |

---

## ⚠️ Analysis of Results

Despite using advanced architectures, classification accuracy remained low.

### Main reasons:

* low dataset diversity
* repeated defect locations
* overfitting to background patterns

This confirms that:
👉 model architecture was NOT the main limitation
👉 dataset quality was critical 

---

## 🔄 Transition to Object Detection (YOLOv8)

Due to poor classification performance, the approach was changed to **object detection**.

YOLOv8 allows:

* defect localization
* detection of multiple defects
* better generalization

---

## 📈 YOLOv8 Results

* mAP@0.5: **76.4%**
* mAP@0.5:0.95: **44.1%**
* Precision: **78.3%**
* Recall: **68.2%**
* F1-score: **0.74**

👉 Significant improvement compared to classification models 

---

## 🧠 Key Insight

The project demonstrates that:

> For PCB defect detection, **object detection approaches (YOLO)** are significantly more effective than pure classification models.

---

## 🖼️ Example Result

*(add your image here)*

```markdown
![YOLO Detection](images/yolo-result.png)
```

---

## 🖥️ Application

A GUI application was developed in Python:

### Features:

* load trained model (.pth)
* upload PCB image
* classify defect type
* detect defect location
* visualize results

---

## 🏭 Practical Applications

* Automated Visual Inspection (AVI)
* PCB manufacturing quality control
* defect detection before assembly
* reduction of faulty products

---

## 🚀 Future Work

* increase dataset diversity
* apply data augmentation
* use segmentation models
* test larger YOLO architectures (YOLOv8-s, YOLOv8-m)

---

## 👨‍💻 Authors

Hubert Jabłoński
Jakub Czekaj

AGH University of Science and Technology
Kraków, Poland

