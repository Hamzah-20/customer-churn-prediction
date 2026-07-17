# Explainable Customer Churn Prediction System using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)

> AI-powered customer churn prediction platform with explainable machine learning and business intelligence analytics.

---

## Overview

**Live Project Repository :** [Churn prediction](https://github.com/Hamzah-20/customer-churn-prediction)

The Explainable Customer Churn Prediction System is a full-stack machine learning platform designed to predict telecom customer churn and provide actionable business insights for customer retention strategies.

The project combines:

- Advanced machine learning pipelines
- Explainable AI techniques
- Dynamic prediction interfaces
- Batch customer analysis
- Business intelligence dashboards

The goal is not only to predict churn, but also to explain _why_ customers are likely to leave and recommend retention strategies.

---

## Key Features

### AI & Machine Learning

- XGBoost (Final Selected Model)
- Random Forest
- Gradient Boosting
- Logistic Regression
- Feature Selection using Mutual Information & L1 Regularization
- Class Imbalance Handling using SMOTETomek
- Advanced Feature Engineering
- Leakage-Free Repeated Stratified Cross Validation

### Explainable AI (XAI)

- SHAP Summary Analysis
- SHAP Feature Importance
- SHAP Explainability Analysis
- Feature Importance Visualization
- Root Cause Analytics

### Web Application

- Dynamic form generation
- Real-time churn prediction
- Batch CSV prediction
- Interactive analytics dashboard
- Responsive UI/UX

### Business Intelligence

- Customer risk segmentation
- Retention recommendations
- Contract impact analysis
- Service adoption insights
- Price sensitivity analysis

---

## Machine Learning Pipeline

The project follows a professional machine learning workflow:

1. Data Cleaning
2. Feature Engineering
3. One-Hot Encoding
4. Train/Test Split
5. SMOTETomek Resampling
6. Feature Selection
7. Model Training
8. Model Comparison
9. Leakage-Free Repeated Stratified Cross Validation
10. Model Selection
11. SHAP Explainability
12. Deployment using Flask

---

## System Architecture

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
One-Hot Encoding
   ↓
Train/Test Split
   ↓
SMOTETomek Resampling
   ↓
Feature Selection
   ↓
Model Training
   ↓
Leakage-Free Cross Validation
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Flask Deployment
```

---

## Scientific Feature Engineering

The feature engineering process was inspired by established business and behavioral theories:

- Customer Lifecycle Theory
- Price Elasticity Theory
- Service Bundling Theory
- Customer Retention Analytics

Examples of engineered features:

- tenure behavior analysis
- pricing sensitivity metrics
- high-risk customer indicators
- service adoption patterns
- contract behavior indicators

---

## Data Leakage Prevention

To ensure reliable evaluation and realistic model performance estimation, all preprocessing steps were performed exclusively on training data during cross-validation.

For each fold:

1. SMOTETomek resampling was applied only to the training split.
2. Feature selection was performed using training data only.
3. Models were trained on the processed training fold.
4. Evaluation was conducted on unseen validation data.

This workflow prevents information leakage and provides a more trustworthy estimate of real-world model performance.

---

## Model Performance

### Why Recall Matters More Than Accuracy?

In churn prediction, missing a customer who is likely to leave is often more expensive than contacting a customer who would stay.

- False Negative → Lost customer and lost revenue
- False Positive → Retention offer with relatively low cost

Therefore, Recall was prioritized as a key evaluation metric.

### Final Test Performance

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 78.32%     | 58.19%     | 65.51%     | 61.64%     | 83.59%     |
| Random Forest       | 77.11%     | 55.73%     | 67.65%     | 61.11%     | 83.74%     |
| Gradient Boosting   | 77.61%     | 57.70%     | 59.09%     | 58.39%     | 82.59%     |
| **XGBoost**         | **74.20%** | **50.95%** | **79.14%** | **61.99%** | **83.21%** |

### Leakage-Free Cross-Validation Results

| Model               | Mean F1 | Std Dev |
| ------------------- | ------- | ------- |
| Logistic Regression | 63.03%  | ±1.22%  |
| Random Forest       | 63.40%  | ±1.37%  |
| Gradient Boosting   | 60.10%  | ±1.67%  |
| XGBoost             | 62.79%  | ±1.17%  |

The reported cross-validation results were obtained using a leakage-free repeated stratified cross-validation workflow, where resampling and feature selection were performed independently within each fold.

### Selected Model

**XGBoost** was selected as the final production model because it achieved the highest F1-score on the held-out test set while also providing the strongest churn detection capability (Recall = 79.14%).

The final model demonstrated stable performance between the test set and cross-validation results, indicating reliable generalization and effective prevention of data leakage.

---

## Explainable AI Visualizations

The platform generates:

- SHAP Summary Plots
- SHAP Importance Charts
- Model Explainability Visualizations
- Confusion Matrix
- Customer Risk Analytics

These visualizations help explain model decisions and improve business trust in AI predictions.

---

## Technologies Used

### Backend

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- XGBoost (imbalanced-learn (SMOTETomek))

### Machine Learning

- Random Forest
- Gradient Boosting
- XGBoost
- Logistic Regression
- SHAP
- SMOTETomek

### Frontend

- HTML5
- CSS3
- JavaScript

### Visualization

- Matplotlib
- Seaborn

---

## Project Structure

```bash
customer-churn-prediction/
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── full_pipeline.pkl
│   ├── selected_features.pkl
│   ├── all_feature_names.pkl
│   └── encoding_info.pkl
│
├── static/
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── confusion_matrix.png
│   └── ...
│
└── templates/
    └── index.html
```

---

## Dataset

Telco Customer Churn Dataset from Kaggle:

- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Links

- GitHub Repository: [customer-churn-prediction](https://github.com/Hamzah-20/customer-churn-prediction)

---

## How Batch Prediction Works

1. Upload a CSV file containing customer information.
2. The system preprocesses the data automatically.
3. The trained ML model predicts churn probability for each customer.
4. Results are displayed with churn risk insights and analytics.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Hamzah-20/customer-churn-prediction.git
cd customer-churn-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000
```

---

## Application Preview

### Dashboard

<img width="1919" height="952" alt="image" src="https://github.com/user-attachments/assets/c34dc0dc-162e-4848-9fdb-0eedb6b095fd" />

### SHAP Explainability

<img width="1350" height="2240" alt="127 0 0 1_5000_" src="https://github.com/user-attachments/assets/b8bd082e-2318-4eb6-bfd2-e44c3ccb4272" />

### Confusion Matrix

<img width="1096" height="880" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c26aa9e5-7144-445a-8888-f5613f2ca8b9" />

### Batch Prediction

<img width="1350" height="2399" alt="127 0 0 1_5000_ (1)" src="https://github.com/user-attachments/assets/9186ed9d-79c0-4e55-99fc-a3d259d430d8" />

---

## Research & Academic Contribution

This project was developed with a strong focus on:

- Preventing data leakage
- Explainable AI
- Business-oriented machine learning
- Real-world deployment architecture
- Professional ML engineering practices

The system demonstrates how interpretable machine learning can improve customer retention strategies in the telecommunications industry.

---

## Important Engineering Practices

- Data leakage prevention using proper train/test separation
- Feature selection applied only on training data
- Pipeline-based preprocessing and training
- Baseline model comparison for academic validity
- Explainable AI integration using SHAP
- Cross-validation stability evaluation
- Clean cross-validation without data leakage

---

## Key Insights

- Customers with month-to-month contracts showed the highest churn probability.
- Fiber optic users had higher churn risk compared to DSL users.
- Long-term customers were significantly less likely to churn.
- Electronic check payment method was strongly associated with customer churn.
- SHAP analysis confirmed that contract type, tenure, monthly charges, and internet service were among the most influential churn drivers.

---

## Future Improvements

- Automated Hyperparameter Optimization using Optuna or Bayesian Optimization
- MLflow Integration for Experiment Tracking and Model Versioning
- End-to-End MLOps Pipeline for Continuous Training and Deployment
- Customer Lifetime Value (CLV) Prediction Integration
- Real-Time Prediction API using FastAPI
- Cloud Deployment on AWS, Azure, or Google Cloud Platform
- Advanced Ensemble Learning Strategies
- Deep Learning-Based Churn Prediction Models
- Automated Data Drift Detection and Monitoring
- Interactive Business Intelligence Dashboard with Real-Time Analytics

---

## Author

Hamzah Albasyouni

AI & Machine Learning Enthusiast  
Focused on Applied AI, Predictive Analytics, and Intelligent Systems

GitHub: [Hamzah-20](https://github.com/Hamzah-20)

---

## License

This project was developed for educational, research, and portfolio purposes.
