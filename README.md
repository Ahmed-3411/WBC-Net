# WBC-Net 🔬

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Framework-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-lightgrey.svg)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-blue.svg)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-lightgrey.svg)](https://scikit-learn.org/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![GPU](https://img.shields.io/badge/GPU-Enabled-green.svg)](https://www.nvidia.com/)


##  Technologies Used

### Core Technologies
- **[Python](https://www.python.org/)** - Primary programming language
- **[TensorFlow](https://tensorflow.org/)** - Deep learning framework
- **[Keras](https://keras.io/)** - Neural network API
- **[Google Colab](https://colab.research.google.com/)** - Development environment

### Data Processing & Model
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[OpenCV](https://opencv.org/)** - Image processing
- **[Matplotlib](https://matplotlib.org/)** - Visualization
- **[Scikit-learn](https://scikit-learn.org/)** - Model evaluation

##  Features

- **Advanced Image Processing**
  - Automatic image resizing (128x128)
  - Pixel normalization
  - Data augmentation with rotations and flips
  - Balanced class distribution

- **Robust CNN Architecture**
  - 4-stage convolutional network
  - Batch normalization
  - Dropout regularization
  - MaxPooling for feature selection

- **Performance Optimization**
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
  - GPU acceleration

- **Comprehensive Evaluation**
  - Confusion matrix analysis
  - ROC curve generation
  - Per-class accuracy metrics
  - Real-time prediction visualization

##  Model Architecture

```
Input (128x128x3)
│
├── Conv Block 1 (64 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Conv Block 2 (128 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Conv Block 3 (256 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Conv Block 4 (512 filters) → BatchNorm → MaxPool → Dropout(0.25)
│
├── Dense(1024) → Dropout(0.5)
│
├── Dense(512) → Dropout(0.5)
│
└── Output(5) → Softmax
```

##  Dataset

- **Source**: [White Blood Cells Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- **Classes**: 5 WBC types
  - Neutrophils
  - Eosinophils
  - Basophils
  - Monocytes
  - Lymphocytes
- **Volume**: 14,514 images
  - Training: 8,140 (80%)
  - Validation: 2,035 (20%)
  - Test: 4,339

## 📈 Performance

- **Confusion Matrix**  
  ![Confusion Matrix](figures/Confusion%20Matrix.png)

- **Class Distribution**  
  ![Feature Heatmap](figures/Training%20Data%20Class%20Distribution.png)

- **Predicted class Neutrophil**  
  ![Operating Characteristic](figures/Predicted%20class%20Neutrophil.png)

  - **Visualizing model predictions**  
  ![Operating Characteristic](figures/Visualizing%20model%20predictions.png)
---

