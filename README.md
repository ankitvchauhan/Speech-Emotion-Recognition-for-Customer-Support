# 🎤 Speech Emotion Recognition using CNN

## 📌 Project Overview

This project aims to detect human emotions from speech audio using Deep Learning.
We use MFCC-based feature extraction and a Convolutional Neural Network (CNN) to classify emotions.

---

## 🎯 Objectives

* Extract meaningful audio features (MFCC, Delta, Delta-Delta)
* Train a CNN model for emotion classification
* Evaluate model performance on unseen data
* Deploy a simple Streamlit-based interface

---

## 📂 Dataset

* Dataset Used: **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)**

### Emotions Covered:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

---

## ⚙️ Methodology

### 1. Feature Extraction

* MFCC (Mel-Frequency Cepstral Coefficients)
* Delta Features
* Delta-Delta Features

### 2. Preprocessing

* Padding audio sequences to fixed length (120)
* Reshaping data for CNN input

### 3. Model Architecture

* Conv2D Layers
* MaxPooling Layers
* Fully Connected Dense Layers
* Dropout for regularization

### 4. Training

* Loss Function: Sparse Categorical Crossentropy
* Optimizer: Adam
* Epochs: 30
* Batch Size: 32

---

## 📊 Results

* Training Accuracy: ~95–100%
* Validation Accuracy: **~58–62%**
* Observed mild overfitting (common in Speech Emotion Recognition tasks)

---

## 🖥️ Project Structure

ser-project-final/
│
├── data/
│   └── ravdess/
│
├── src/
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│
├── app/
│   └── app.py
│
├── models/
│   ├── model.keras
│   └── encoder.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

---

## 🚀 How to Run

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Train Model

python src/train.py

### 3. Run Streamlit App

streamlit run app/app.py

---

## 💡 Future Improvements

* Data augmentation for better generalization
* Hyperparameter tuning
* Use of advanced architectures (LSTM / CNN-LSTM)
* Real-time emotion detection

---

## 🎓 Conclusion

The project successfully demonstrates emotion recognition using deep learning.
While accuracy is moderate, the system is stable, explainable, and deployable.

---

## 👨‍💻 Author

Ankit Chauhan

---
