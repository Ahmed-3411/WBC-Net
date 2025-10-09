# 🔬 White Blood Cell Classification using CNN

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Framework-red.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A deep learning project implementing a Convolutional Neural Network (CNN) for automated classification of white blood cells from microscopic images, achieving 95% accuracy across five cell types.

**Author:** Muhammad Husnain Ali

## 🛠️ Technologies Used

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

## 🚀 Features

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

## 🏗️ Model Architecture

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

## 📊 Dataset

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

| Metric | Score |
|--------|--------|
| Training Accuracy | 93.73% |
| Validation Accuracy | 94.94% |
| Test Accuracy | 95.00% |
| Inference Time | <100ms |

## 🚀 Quick Start

### 1. Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M-Husnain-Ali/White-Blood-Cell-Classification/blob/main/code.ipynb)

### 2. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Install Dependencies
```python
!pip install -r requirements.txt
```

### 4. Download Dataset
```python
!kaggle datasets download -d paultimothymooney/blood-cells
!unzip blood-cells.zip
```

### 5. Run Training
```python
!python train.py --epochs 10 --batch_size 32
```

## 📦 Requirements

```
tensorflow>=2.8.0
numpy>=1.19.5
opencv-python>=4.5.5
matplotlib>=3.4.3
scikit-learn>=1.0.2
```

## 💡 Usage Tips

### Training
- Use GPU runtime in Colab
- Enable data augmentation
- Start with provided hyperparameters
- Monitor validation metrics

### Prediction
```python
model = load_model('wbc_classifier.h5')
prediction = model.predict(image)
```

### Evaluation
```python
from evaluation import evaluate_model
metrics = evaluate_model(model, test_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

For questions and support:
- Open an [Issue](https://github.com/M-Husnain-Ali/White-Blood-Cell-Classification/issues)
- Contact: m.husnainali.work@gmail.com

## 🙏 Acknowledgments

- Kaggle for the comprehensive dataset
- TensorFlow team for the framework
- Google Colab for free GPU resources
- All contributors to this project

---
Made with ❤️ by Muhammad Husnain Ali
