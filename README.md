# Chronos AI — Customer Churn Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-red)


> AI-powered customer churn prediction platform with explainable machine learning and business intelligence analytics.

---

## Overview

**Live Project Repository :**  [Churn prediction](https://github.com/Hamzah-20/customer-churn-prediction)

Chronos AI is a full-stack machine learning platform designed to predict telecom customer churn and provide actionable business insights for customer retention strategies.

The project combines:
- Advanced machine learning pipelines
- Explainable AI techniques
- Dynamic prediction interfaces
- Batch customer analysis
- Business intelligence dashboards

The goal is not only to predict churn, but also to explain *why* customers are likely to leave and recommend retention strategies.

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
- Cross-Validation Evaluation

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
9. Cross-Validation Evaluation
10. Model Selection
11. SHAP Explainability
12. Deployment using Flask

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

## Model Performance

### Why Recall Matters More Than Accuracy?

In churn prediction, missing a customer who is likely to leave is often more expensive than contacting a customer who would stay.

* False Negative → Lost customer and lost revenue
* False Positive → Retention offer with relatively low cost

Therefore, Recall was prioritized as a key evaluation metric.

### Final Test Performance

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 78.04%     | 57.87%     | 63.90%     | 60.74%     | 83.53%     |
| Random Forest       | 77.40%     | 56.33%     | 66.58%     | 61.03%     | 83.65%     |
| Gradient Boosting   | 77.90%     | 58.49%     | 58.02%     | 58.26%     | 82.78%     |
| **XGBoost**         | **73.77%** | **50.43%** | **78.61%** | **61.44%** | **83.14%** |

### Cross-Validation Results

| Model               | Mean F1 | Std Dev |
| ------------------- | ------- | ------- |
| Logistic Regression | 62.29%  | ±1.45%  |
| Random Forest       | 63.58%  | ±1.54%  |
| Gradient Boosting   | 60.13%  | ±1.85%  |
| XGBoost             | 62.57%  | ±1.17%  |

### Selected Model

**XGBoost** was selected as the final model because it achieved the highest churn detection capability (Recall = 78.61%) while maintaining strong ROC-AUC performance.

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
├── static/
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── confusion_matrix.png
│   └── ...
│
├── templates/
│   └── index.html
│
├── full_pipeline.pkl
├── selected_features.pkl
├── all_feature_names.pkl
├── encoding_info.pkl
├── churn_model.pkl
└── scaler.pkl
```

---

## Dataset

Telco Customer Churn Dataset from Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

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

- Docker deployment
- Cloud hosting (AWS/GCP)
- Deep Learning experimentation
- Real-time API integration
- Advanced hyperparameter optimization
- MLOps pipeline integration

---

## Author

Hamzah Albasyouni

AI & Machine Learning Enthusiast  
Focused on Applied AI, Predictive Analytics, and Intelligent Systems

GitHub: [Hamzah-20](https://github.com/Hamzah-20)

---

## License

This project was developed for educational, research, and portfolio purposes.
