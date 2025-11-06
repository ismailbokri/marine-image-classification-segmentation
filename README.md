# 🌊 Marine Waste Detection

## 🧭 Project Overview

**Marine Waste Detection** is an AI-based project designed to automatically **classify and segment marine debris** from underwater sonar images.  
The main goal is to support ocean cleanup initiatives by detecting and analyzing pollution sources in marine environments.

---

## 🎯 Objectives

### 🔹 1. Waste Classification

The first objective focuses on **classifying the detected debris** based on predefined categories.  
The training dataset was organized into folders labeled by class names.

**Class labels used:**
```python
CLASS_NAMES = [
    'bottle', 'can', 'chain', 'drink-carton', 'hook',
    'propeller', 'shampoo-bottle', 'standing-bottle', 'tire', 'valve'
]
📓 A training notebook is available inside the training/ directory for model reproduction and experimentation.
```
### 🔹 2. Waste Segmentation
The second phase focuses on precise segmentation of marine debris in sonar images.
The original dataset lacked segmentation masks, which are required for training segmentation models.

To overcome this limitation, we used the SAM (Segment Anything Model) to:

Automatically generate high-quality masks for each object.

Avoid manual annotation efforts.

Build a clean dataset suitable for training a segmentation network.

🧠 Both models (classification and segmentation) were trained using TensorFlow.

## ⚙️ Technologies Used
Category	Tools / Frameworks
Programming Language	Python
Deep Learning Framework	TensorFlow
Segmentation Tool	SAM (Segment Anything Model)
Web Deployment	Flask
Visualization / Notebooks	Jupyter Notebook
