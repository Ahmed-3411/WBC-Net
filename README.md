# WBC-Net: A Convolutional Neural Network for Automated White Blood Cell Classification

[![Paper](https://img.shields.io/badge/Paper-TechRxiv-brightgreen)](LINK_TO_TECHRXIV_PAGE)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Keras](https://img.shields.io/badge/API-Keras-red)
![OpenCV](https://img.shields.io/badge/Image%20Processing-OpenCV-lightgrey)

> **WBC-Net** is a deep convolutional neural network for automated classification of white blood cells.  
> **Neutrophils, Lymphocytes, Monocytes, Eosinophils, and Basophils.**

---

## Abstract

White blood cell (WBC) analysis is crucial for early detection of hematological disorders.  
This repository provides the implementation of **WBC-Net**, a convolutional neural network
designed for robust and scalable WBC classification from microscopic images.

The pipeline includes advanced image preprocessing (resizing, normalization, and augmentation),
a four-stage CNN with batch normalization and dropout regularization, and performance optimization
techniques such as early stopping, learning-rate scheduling, and model checkpointing.  
Extensive experiments on a public WBC dataset show that WBC-Net achieves strong performance
across multiple metrics, making it suitable for real-time clinical decision-support applications.

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Dataset](#2-dataset)  
3. [Methodology](#3-methodology)  
4. [Experiments & Results](#4-experiments--results)  
5. [Repository Structure](#5-repository-structure)  
6. [Installation](#6-installation)  
7. [Usage](#7-usage)  
8. [Citation](#9-citation)  
9. [Contact](#10-contact)

---

## 1. Project Overview

The goal of **WBC-Net** is to automate multi-class WBC classification using a deep learning pipeline optimized for accuracy, robustness, and reproducibility.

### Key Contributions

- A **4-stage CNN architecture** tailored for WBC morphology  
- **Comprehensive preprocessing pipeline** including resizing, augmentation, and normalization  
- **Training optimization**: batch normalization, dropout, learning-rate scheduling  
- **Evaluation suite**: confusion matrix, ROC curves, per-class metrics  
- **GPU-optimized implementation**  

---

## 2. Dataset

- **Source:**  
  [White Blood Cells Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

- **Classes (5):**  
  - Neutrophils  
  - Eosinophils  
  - Basophils  
  - Monocytes  
  - Lymphocytes

- **Total Images:** 14,514  
  - Training: 8,140 (80%)  
  - Validation: 2,035 (20%)  
  - Test: 4,339  

---

## 3. Methodology

### 3.1 Preprocessing Pipeline
- Resize images to **128×128**
- Pixel normalization  
- Augmentation: rotations, flips  
- Addressing class imbalance  

### 3.2 CNN Architecture

```
Input (128x128x3)
├── Conv Block 1 (64 filters) → BatchNorm → MaxPool → Dropout(0.25)
├── Conv Block 2 (128 filters) → BatchNorm → MaxPool → Dropout(0.25)
├── Conv Block 3 (256 filters) → BatchNorm → MaxPool → Dropout(0.25)
├── Conv Block 4 (512 filters) → BatchNorm → MaxPool → Dropout(0.25)
├── Dense(1024) → Dropout(0.5)
├── Dense(512) → Dropout(0.5)
└── Output(5) → Softmax
```

---

## 4. Experiments & Results

### Confusion Matrix
![Confusion Matrix](figures/Confusion%20Matrix.png)

### Class Distribution  
![Training Distribution](figures/Training%20Data%20Class%20Distribution.png)

### Sample Prediction  
![Prediction](figures/Predicted%20class%20Neutrophil.png)

### Visualizing Model Predictions  
![Predictions](figures/Visualizing%20model%20predictions.png)

---

## 5. Repository Structure

```
WBC-Net/
 ├── models/
 ├── figures/
 ├── data/
 ├── notebooks/
 ├── scripts/
 └── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/WBC-Net.git
cd WBC-Net
pip install -r requirements.txt
```

---

## Usage

### Training
```bash
python train.py --dataset data/
```

### Evaluation
```bash
python evaluate.py --weights best_model.h5
```

---

## Citation

If you use WBC-Net in your research, please cite:

```
Ahmed Talaat Mersal et al.   
WBC-Net: A Convolutional Neural Network for Automated White Blood Cell Classification.  
TechRxiv, 2025.  
DOI: 10.36227/techrxiv.176282158.83951263/v1

```

---

## Contact

**Ahmed Talaat**  
📧 ahmed.mmt3411@gmail.com  
🔗 GitHub: @Ahmed-3411
