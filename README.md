# 💼 Employee Salary Prediction App

A machine learning web application that predicts whether an individual earns more than 50K per year based on demographic and professional attributes. Built using Random Forest Classifier and deployed via Streamlit with SHAP and ROC visualizations.

---

## 📊 Project Overview

This project uses the [Adult Income Dataset] (also known as the Census Income dataset) to predict employee salary categories (`<=50K` or `>50K`).

We have built:
- A **Streamlit Web App** for real-time single and batch predictions
- **Model Explainability** using SHAP values
- **Performance Visualization** via ROC curves

---

## 📁 Dataset Features

The dataset includes demographic, education, occupation, and financial variables:

- Age  
- Workclass  
- Education & Educational-num  
- Marital-status  
- Occupation  
- Relationship  
- Race  
- Gender  
- Capital-gain / Capital-loss  
- Hours-per-week  
- Native-country  
- Income (Target Variable)

---

## 🧠 Machine Learning Approach

- **Model Used**: Random Forest Classifier
- **Preprocessing**:
  - Label Encoding for categorical features
  - Standard Scaling of numerical features
  - Removed irrelevant features like `fnlwgt`
- **Evaluation Metrics**:
  - Accuracy
  - ROC AUC Score
  - Confusion Matrix
  - SHAP Explainability

---

## 💻 Web App Features

### 🎯 Single Prediction
- Interactive sliders and dropdowns to enter employee details
- Shows prediction (`>50K` or `<=50K`) with probability
- SHAP waterfall plot to explain feature contributions

### 📁 Batch Prediction
- Upload a CSV of employee records
- Returns predictions + probabilities
- Option to download results as CSV
- Displays ROC Curve if actual labels are provided

---

## 🧪 Installation

```bash
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
pip install -r requirements.txt

Developed by [ Sanyog Varun Panda ]
Feel free to connect or contribute!
