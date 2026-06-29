# 🛰️ OtterMap Turf Segmentation using SegFormer-B2

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end semantic segmentation pipeline for **automatic turf/grass detection** from high-resolution aerial orthophotos. The project leverages **SegFormer-B2** with transfer learning to generate accurate segmentation masks and GIS-compatible outputs such as **GeoJSON** for downstream geospatial workflows.

---

# 📌 Overview

This project was developed for the **OtterMap Open Vision / ML Engineer Technical Challenge**.

The objective is to train a semantic segmentation model capable of identifying **turf/grass regions** from aerial imagery and generalizing to previously unseen geographic locations.

The complete workflow includes:

* Polygon annotation preprocessing
* Raster mask generation
* Patch creation
* SegFormer-B2 fine-tuning
* Batch inference
* Polygonization
* GeoJSON generation
* External validation on unseen aerial imagery

---

# ✨ Features

* Semantic Segmentation using **SegFormer-B2**
* Transfer Learning with Hugging Face Transformers
* Automatic raster mask generation
* Patch-based training pipeline
* Mixed Precision (AMP) Training
* Batch inference on aerial imagery
* GeoJSON generation
* GIS-compatible polygon outputs
* External validation on unseen imagery

---

# 📂 Project Structure

```text
ottermap-segformer-project/
│
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
│
├── configs/
├── data/
├── datasets/
├── inference/
│   ├── inference.py
│   └── polygonize.py
├── models/
├── training/
├── utils/
├── outputs/
│   ├── predictions/
│   ├── overlays/
│   ├── geojson/
│   └── external_validation/
├── logs/
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🗂 Dataset Preparation

Training data consists of high-resolution aerial orthophotos and polygon annotations representing turf regions.

### Preprocessing Pipeline

```text
Polygon Labels
      │
      ▼
Raster Masks
      │
      ▼
Patch Generation
      │
      ▼
Train / Validation Split
      │
      ▼
Data Augmentation
```

---

# 🧠 Model Architecture

| Component         | Details                   |
| ----------------- | ------------------------- |
| Architecture      | SegFormer-B2              |
| Framework         | PyTorch                   |
| Backbone          | NVIDIA SegFormer-B2       |
| Transfer Learning | Hugging Face Transformers |
| Loss Function     | Cross Entropy Loss        |
| Optimizer         | AdamW                     |
| Scheduler         | Cosine Annealing LR       |
| Mixed Precision   | Enabled                   |

---

# 📈 Training Results

The model was trained for **40 epochs** and the best checkpoint was automatically saved.

## Validation Performance

| Metric                  |      Value |
| ----------------------- | ---------: |
| Training Epochs         |     **40** |
| Best Validation IoU     | **92.72%** |
| Validation IoU (Logged) | **92.03%** |
| Dice Score              | **95.85%** |
| Precision               | **96.13%** |
| Recall                  | **95.57%** |
| F1 Score                | **95.85%** |
| Pixel Accuracy          | **96.05%** |
| Training Loss           | **0.0598** |
| Validation Loss         | **0.0826** |

The model converged successfully with a low validation loss and achieved strong segmentation performance on the validation dataset.

---

# 🌍 External Validation

To evaluate model generalization, inference was performed on **previously unseen aerial imagery** from the **Massachusetts Buildings Dataset**.

The pipeline successfully generated:

* Prediction Masks
* Overlay Visualizations
* GeoJSON Outputs

This demonstrates that the trained model can perform inference on imagery outside the original training distribution while maintaining GIS-compatible outputs.

---

# 🗺 GIS Output Generation

Prediction masks are automatically converted into GIS-ready vector data.

Supported outputs:

* GeoJSON
* Polygon Masks

These outputs can be directly integrated into GIS workflows and mapping software.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mayanksinha-dev/ottermap-segformer-project.git

cd ottermap-segformer-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

Start training:

```bash
python -m training.train
```

The best checkpoint is automatically saved as:

```text
checkpoints/best_model.pth
```

---

# 🔍 Inference

Run inference on a single image:

```bash
python -m inference.inference --image input_image.tif
```

Outputs:

```text
outputs/
├── predictions/
├── overlays/
└── geojson/
```

---

# 📦 Batch Inference

Run inference on an entire folder:

```bash
python batch_inference.py --input_dir ./images
```

Outputs include:

* Prediction Masks
* Overlay Images
* GeoJSON Files

---

# 📊 Sample Outputs

The repository includes:

* Training Predictions
* Validation Predictions
* External Validation Results
* Overlay Visualizations
* GeoJSON Outputs

---

# 📚 Dependencies

* Python 3.10+
* PyTorch
* Transformers
* Rasterio
* OpenCV
* NumPy
* GeoPandas
* Shapely
* Matplotlib

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Model

The trained checkpoint is available on Kaggle:

https://drive.google.com/drive/folders/1JszYZzvzWCK9_3CNROAJOotQhKQGU_SI?usp=sharing

Download `best_model.pth` and place it in:

checkpoints/
    best_model.pth

---

# 🔮 Future Improvements

* Multi-class semantic segmentation
* Larger geographically diverse datasets
* Test-Time Augmentation (TTA)
* ONNX/TensorRT deployment
* Lightweight inference for edge devices
* WebGIS integration
* Active learning pipeline

---

# 🙏 Acknowledgements

* OtterMap
* NVIDIA SegFormer
* Hugging Face Transformers
* PyTorch
* Rasterio
* OpenCV

---

## ⭐ Project Highlights

* ✅ End-to-End Semantic Segmentation Pipeline
* ✅ Transfer Learning using SegFormer-B2
* ✅ Best Validation IoU: **92.72%**
* ✅ Pixel Accuracy: **96.05%**
* ✅ Dice Score: **95.85%**
* ✅ Automatic GeoJSON Generation
* ✅ External Validation on Unseen Aerial Imagery
* ✅ GIS-Compatible Outputs
