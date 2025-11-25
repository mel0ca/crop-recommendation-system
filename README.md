# 🌾 Crop Recommendation System

A machine learning–based system that recommends the most suitable crop for cultivation based on key environmental parameters such as soil nutrients, temperature, humidity, pH, and rainfall. This project helps farmers make data-driven decisions to improve productivity and sustainability.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)

---

## 📘 Project Overview

The objective of this project is to build an intelligent recommendation system that predicts the best crop for a particular set of environmental conditions. Using supervised machine learning algorithms, the system learns patterns from historical agricultural data and outputs the ideal crop for cultivation.

This empowers farmers with:
- Better yield
- Efficient soil and resource usage
- Smart agricultural planning

---

## 📊 Dataset

The dataset **Crop_recommendation.csv** contains:

| Feature | Description |
|--------|-------------|
| **N** | Nitrogen content in soil |
| **P** | Phosphorus content |
| **K** | Potassium content |
| **temperature** | Temperature (°C) |
| **humidity** | Relative humidity (%) |
| **ph** | Soil pH value |
| **rainfall** | Rainfall (mm) |
| **label** | Recommended crop |

### 🔧 Data Preprocessing Steps
1. Load dataset using Pandas  
2. Perform EDA: shape, statistics, duplicates, null check  
3. Separate features (`X`) and target (`y`)  
4. Encode target labels using `LabelEncoder`  
5. Split into Train/Test sets (80/20)  
6. Scale features using `StandardScaler`

---

## 🔍 Methodology

1. **Data Preparation**  
   Cleaning, encoding, scaling, and splitting.

2. **Model Training**  
   Multiple machine learning classification models were trained.

3. **Evaluation**  
   Models were compared using:
   - Accuracy Score  
   - Classification Report  
   - Confusion Matrix (for best model)

---

## 🤖 Models Used

The following classification models were trained:

- Logistic Regression  
- Gaussian Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  
- Decision Tree  
- Extra Tree Classifier  
- Gradient Boosting Classifier  
- AdaBoost Classifier  
- Bagging Classifier  
- K-Nearest Neighbors (KNN)

---

## 🏆 Results

### Model Accuracies:
| Model | Accuracy |
|-------|----------|
| Logistic Regression | **0.9636** |
| Gaussian NB | **0.9955** |
| SVM | **0.9682** |
| Random Forest | **0.9932** |
| Decision Tree | **0.9886** |
| Extra Tree | **0.8773** |
| Gradient Boosting | **0.9818** |
| AdaBoost | **0.1455** |
| Bagging | **0.9886** |
| KNN | **0.9568** |

### 🥇 Best Model: **Gaussian Naive Bayes (Accuracy: 0.9955)**

A confusion matrix was also generated to visualize prediction accuracy across all crop types.

---

## ⚙ Installation

Install dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn