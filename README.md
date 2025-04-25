# 🧠 Alzheimer's Disease Detection Using Machine Learning

This project demonstrates the power of **Machine Learning** in healthcare by detecting different stages of Alzheimer's Disease using image data. Leveraging **Convolutional Neural Networks (CNN)** and **transfer learning** with **InceptionV3**, I developed a robust classification model that achieved **97.44% accuracy**.

---

## 🚀 Project Overview
Alzheimer’s Disease affects millions worldwide. Early and accurate diagnosis is critical. This project applies **machine learning techniques** to classify brain MRI images into four stages:
- **Non-Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

### 🔍 Objective:
To build an efficient machine learning pipeline for **image classification**, handling **imbalanced data**, and delivering high performance on medical image datasets.

---

## 💡 Key Highlights
- Developed a **deep learning model** using **TensorFlow** and **Keras**.
- Applied **image augmentation** and **ADASYN oversampling** to improve data balance.
- Fine-tuned **InceptionV3** (a pre-trained CNN) for feature extraction.
- Evaluated using **confusion matrix**, **classification report**, and **visual analytics**.
- Achieved **97.44% accuracy** on the test dataset.

---

## ⚙️ Machine Learning Workflow
1. **Data Preprocessing**:
   - Image resizing, normalization, augmentation (zoom, brightness, flipping).
   - Balanced the dataset using **ADASYN**.

2. **Model Development**:
   - Transfer Learning with **InceptionV3** (pre-trained on ImageNet).
   - Custom dense layers with **Dropout** and **L2 regularization** to prevent overfitting.
   - Optimized with **Adam** optimizer and **early stopping**.

3. **Evaluation**:
   - Metrics: **Accuracy**, **Loss**, **Balanced Accuracy**, **Matthews Correlation Coefficient**.
   - Visualized class distribution and model performance.

---

## 📊 Tools & Technologies
- **Python**, **NumPy**, **Pandas**
- **TensorFlow**, **Keras**
- **Scikit-learn**: Evaluation Metrics
- **Imbalanced-learn**: ADASYN for data balancing
- **Matplotlib**, **Seaborn**: Data visualization
- **Google Colab**: Model training environment

---

## 📈 Visual Insights
- **Training & Validation Accuracy**
- **Loss Curves**
- **Confusion Matrix**

