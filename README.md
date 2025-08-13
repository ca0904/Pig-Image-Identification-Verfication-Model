# Pig Image Identification and Verification Model

A comprehensive deep learning solution for **pig identification** and **verification** using computer vision techniques. This Bachelor's Thesis Project (BTP) implements a state-of-the-art biometric system for individual pig recognition with both closed-set identification and open-set recognition capabilities.

## 🎯 Project Overview

This project develops a robust pig identification system that can:

- **Identify individual pigs** from images (closed-set classification)
- **Detect unknown/new pigs** not seen during training (open-set recognition)
- **Verify pig identity** through pairwise similarity matching
- **Handle real-world scenarios** with data augmentation and quality filtering

## 🏗️ Architecture & Methodology

### Core Components

- **Feature Extraction**: EfficientNet-B0 backbone with multi-layer perceptron head
- **Metric Learning**: ArcFace loss with center loss regularization
- **Open-Set Recognition**: Prototype-based similarity scoring with adaptive thresholds
- **Verification System**: Pairwise similarity matching with Equal Error Rate optimization

### Key Features

- **Configurable feature embeddings** (256/384/512 dimensions)
- **Multi-layer feature projection head** with 3 layers of 128 hidden units
- **Robust data preprocessing** with blur detection and dimension filtering
- **5-fold stratified cross-validation** for comprehensive evaluation
- **Hyperparameter optimization** using Optuna TPE sampler
- **Comprehensive evaluation metrics** including AUROC, AUPR, and EER

## 📊 Performance Results

### Cross-Validation Results (5-Fold)

| Metric | Mean ± Std |
|--------|------------|
| **Identification Accuracy** | 99.44% ± 0.24% |
| **Open-Set AUROC** | 99.95% ± 0.04% |
| **Verification AUROC** | 98.47% ± 0.26% |
| **Verification EER** | 6.36% ± 0.25% |
| **Open-Set Precision** | 99.66% ± 0.15% |
| **Open-Set Recall** | 99.45% ± 0.24% |

### Model Configuration

- **Feature Dimensions**: 384 (configurable: 256/384/512)
- **MLP Head**: 3 layers × 128 hidden units
- **Activation Function**: GELU
- **Normalization**: Batch Normalization
- **Dropout Rate**: 0.46
- **ArcFace Scaling**: 46
- **Angular Margin**: 0.31
- **Optimizer**: AdamW (LR: 2.67e-4)

## 🚀 Quick Start

> **⚠️ Important:** Complete execution may take up to **8 days** due to extensive cross-validation.

### 1. Data Preparation

```bash
python code/data_preparation.py
```

Prepares the dataset, creates train/validation/test splits, and generates distribution visualizations.

### 2. Cross-Validation Training

Run all fold scripts (each takes ~1.5 days):

```bash
python code/cv_fold_1.py
python code/cv_fold_2.py
python code/cv_fold_3.py
python code/cv_fold_4.py
python code/cv_fold_5.py
```

### 3. Final Model Training & Evaluation

```bash
python code/model_training.py
```

### 4. Interactive Analysis

```bash
jupyter notebook id-and-verif-model.ipynb
```

> **Note:** `utils.py` is a helper module - do not run directly.

## 📁 Project Structure

```text
├── 📓 id-and-verif-model.ipynb    # Complete interactive analysis notebook
├── 📝 README.md                   # This file
├── 💻 code/                       # Source code
│   ├── data_preparation.py        # Dataset preprocessing & splits
│   ├── model_training.py          # Final model training & evaluation  
│   ├── cv_fold_*.py              # Cross-validation fold scripts (1-5)
│   ├── utils.py                  # Core utilities & model definitions
│   └── jobSchedule/              # SLURM job scheduling scripts
├── 🎯 weights/                    # Model outputs & results
│   ├── *.pt                      # Trained model weights
│   ├── *.csv                     # Performance metrics & results
│   ├── *.json                    # Final results & best parameters
│   ├── *.png                     # Visualizations & plots
│   └── *.pkl                     # Preprocessed data & trained models
├── 📊 ppt/                        # Project presentation
│   ├── Pig Id-Verif Model.pptx   # PowerPoint presentation
│   └── Pig Id-Verif Model.pdf    # PDF version
└── 📄 report/                     # Project documentation
    ├── BTP Project Report.docx    # Detailed project report
    └── BTP Project Report.pdf     # PDF version
```

## 🔬 Technical Implementation

### Data Pipeline

1. **Image Filtering**: Removes blurry images and enforces dimension constraints
2. **Dataset Splitting**: 80/20 known/unknown split with stratified sampling
3. **Data Augmentation**: Rotation, flipping, color jittering, and random crops
4. **Quality Control**: Automatic blur detection and size validation

### Model Architecture

- **Backbone**: EfficientNet-B0 with pretrained weights
- **Feature Head**: 3-layer MLP (hidden: 128, output: configurable feature dims)
- **Loss Function**: Combined ArcFace + Center Loss
- **Optimization**: AdamW with CosineAnnealingWarmRestarts scheduler

### Evaluation Framework

- **Closed-Set Identification**: Standard classification metrics
- **Open-Set Recognition**: AUROC, AUPR, threshold optimization
- **Verification**: ROC analysis, EER computation, similarity distributions

## 📈 Key Innovations

1. **Hybrid Loss Function**: Combines ArcFace for discrimination with Center Loss for compactness
2. **Multi-Scale Evaluation**: Tests both identification and verification capabilities
3. **Robust Open-Set Handling**: Prototype-based unknown detection with adaptive thresholds
4. **Comprehensive Validation**: 5-fold cross-validation with hyperparameter optimization

## 🛠️ Dependencies

- **Deep Learning**: PyTorch, torchvision, timm
- **Computer Vision**: OpenCV, PIL
- **Machine Learning**: scikit-learn, optuna
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **Evaluation**: Custom metrics for biometric evaluation

## 📚 Documentation

- **Detailed Report**: `report/BTP Project Report.pdf`
- **Presentation**: `ppt/Pig Id-Verif Model.pdf`
- **Interactive Analysis**: `id-and-verif-model.ipynb`
- **Results Visualization**: Multiple plots in `weights/` directory

## 🎯 Use Cases

- **Livestock Management**: Individual pig tracking and monitoring
- **Research Applications**: Behavioral studies requiring individual identification
- **Farm Automation**: Automated feeding and health monitoring systems
- **Biometric Research**: Novel applications in animal biometrics

---

**Institution**: Indian Institute of Technology (BHU) Varanasi
**Year**: 2025

> For detailed methodology, results analysis, and technical implementation, refer to the comprehensive project report and interactive notebook.
