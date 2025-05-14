# Project-Data-Mining
# 🧠 Mental Health Prediction using Machine Learning

A data mining project to predict depression risk based on mental health survey data.

---

## 🧠 Overview

This project aims to build a classification model that can predict whether a person is experiencing depression using the **"Exploring Mental Health"** dataset from Kaggle. 

Using machine learning algorithms and the **CRISP-DM** methodology, we explore data, prepare features, train models, evaluate them, and finally deploy the best-performing model using **Streamlit**.

---

## 🔍 1. Business Understanding

Depression is a common mental health issue that affects well-being and productivity. This project seeks to:

- ✅ Build a predictive model to detect potential depression cases  
- ✅ Understand key factors contributing to mental health conditions  

---

## 📊 2. Data Understanding

The dataset includes:

- Demographic information  
- Academic/work pressures  
- Dietary habits  
- Family mental illness history  
- Sleep patterns  
- Other relevant attributes  

**Steps Taken:**

- Inspected structure and types of features  
- Visualized numerical and categorical distributions  
- Analyzed correlation with the target feature `Depression`  

---

## 🧹 3. Data Preparation

### 🔧 Data Cleaning

- Removed duplicates  
- Imputed missing values with **median** (numerical) and **mode** (categorical)  
- Handled outliers based on domain logic  

### 🏗 Feature Engineering

- Normalized features using `MinMaxScaler`  
- Converted ordinal features (e.g., Sleep Duration, Dietary Habits) to numeric  
- Created new feature:  
  ```python
  Stress_Score = Financial_Stress + Work_Pressure
### 🔤 Encoding

- Applied **LabelEncoder** and `.cat.codes` to categorical features  
- Encoded **binary target variable**  
- Encoded **role** (Working Professional or Student)

---

## 🧠 4. Modeling

### Models Used:
- **Decision Tree (C4.5-style)** using entropy  
- **XGBoost** (default parameters)  
- **XGBoost + GridSearchCV** (hyperparameter tuning)

### Train/Test Split:
- **80/20 split**

---

## 📈 5. Evaluation

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC

| Model                    | Accuracy | Precision | Recall | F1-Score | AUC  |
|-------------------------|----------|-----------|--------|----------|------|
| C4.5 Decision Tree      | 92.71%   | 79.27%    | 81.53% | 80.38%   | -    |
| XGBoost                 | 93.24%   | 82.21%    | 80.50% | 81.34%   | -    |
| XGBoost + GridSearchCV  | 93.90%   | 84.50%    | 81.66% | 83.06%   | 0.97 |

---

## 🚀 6. Deployment

### Local Deployment:
- Backend: **Flask** (`app.py`)  
- HTML form for user input and prediction  
- Styled with `style.css`

### Online Deployment:
- **Streamlit app** using `streamlit_app.py`  
- Requirements listed in `requirements.txt`
