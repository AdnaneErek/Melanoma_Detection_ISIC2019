# Melanoma_Detection_ISIC2019
 
**Image + Metadata Fusion with ResNet-50 & Lightning**

This repository contains a multimodal deep learning pipeline for the **ISIC 2019** melanoma classification challenge.  
The model integrates:

- **Dermoscopic images** (processed to 224×224, hair-removed)
- **Patient metadata** (age, sex, anatomical site)
- **ResNet-50 visual backbone**
- **Metadata MLP**
- **Late fusion classifier**

Training is performed with **PyTorch Lightning (DDP)** on multi-GPU nodes.

---

## Features

### ✔ Multimodal Architecture
- ResNet-50 (ImageNet pretrained) for image features  
- MLP for metadata encoding  
- Joint fusion layer (2048 + 128 → 256 → 8 classes)

### ✔ Two-Stage Training Strategy
1. **Warm-up** (epochs 0–2): backbone frozen  
2. **Fine-tuning** (epoch ≥3): full model optimized (layer-wise LR decay)

### ✔ Class-Weighted Loss  
Handles strong class imbalance in ISIC.

### ✔ Full Metrics Suite  
- Accuracy  
- Macro F1  
- Weighted F1  
- Per-class precision/recall/F1  
- AUROC  
- Confusion matrix logged to TensorBoard

### ✔ Distributed Training (DDP)
Compatible with SLURM (`srun python …`).

### ✔ Grad-CAM Visualization  
Side-by-side (original + heatmap) for model interpretability.

---

##  Repository Structure
```
Melanoma_Detection_ISIC2019/
│
├── src/
│ ├── train/
│ │ ├── train.py # Training script
│ │ ├── lightning_module.py # Lightning module
│ │ ├── gradcam.py # Grad-CAM script
│ │
│ ├── models/
│ │ └── combined_model.py # ResNet + MLP fusion
│ │
│ └── dataset/
│ └── isic_dataset.py # Image+metadata dataset
│
├── processed/
│ ├── images_224_nohair/ # Preprocessed images
│ └── splits/
│ ├── train.csv
│ └── val.csv
│
├── checkpoints/ # Best and final models
├── logs/ # TensorBoard + CSV logs
├── requirements.txt
└── README.md
```

## Training

---

### **Single node, 4 GPUs (DDP)**

```bash
python src/train/train.py
```

## Installation
```
git clone https://github.com/AdnaneErek/Melanoma_Detection_ISIC2019/
cd Melanoma_Detection_ISIC2019
pip install -r requirements.txt
```
