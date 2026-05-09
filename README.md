# Chronos AI — Customer Churn Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-red)


> AI-powered customer churn prediction platform with explainable machine learning and business intelligence analytics.

---

## Overview

** Live Project Repository:** [github.com/Hamzah-20/customer-churn-prediction](https://github.com/Hamzah-20/customer-churn-prediction)

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
- Ensemble Learning Models
- Random Forest
- Gradient Boosting
- Voting Classifier
- Logistic Regression Baseline
- Feature Selection using Mutual Information
- Class Imbalance Handling using SMOTETomek
- Advanced Feature Engineering

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
8. Ensemble Learning
9. Model Evaluation
10. SHAP Explainability
11. Deployment using Flask

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
In churn prediction, **missing a churning customer (False Negative) is more costly**
than a false alarm (False Positive).

- Missing a churner = losing a customer forever
- False alarm = unnecessary retention offer (small cost)

Therefore, **Recall was prioritized** as the primary model selection metric.

---

### Final Test Performance (on original imbalanced data)

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 78.39%   | 58.14%    | 66.84% | 62.19%   | 83.57%  |
| **Random Forest**  | **77.61%** | **56.51%** | **68.45%** | **61.91%** | **83.62%** |
| Gradient Boosting   | 78.25%   | 58.95%    | 59.89% | 59.42%   | 82.95%  |
| Voting Ensemble     | 78.25%   | 58.25%    | 64.17% | 61.07%   | 83.46%  |

>  **Best Model Selected: Random Forest**
> Reason: Highest Recall (68.45%) and ROC-AUC (83.62%) —
> ensuring maximum detection of at-risk customers.

---

### Cross-Validation Stability (on resampled training data)

| Model               | Mean F1 | Std Dev | 95% Confidence Interval |
| ------------------- | ------- | ------- | ----------------------- |
| Logistic Regression | 86.45%  | ±1.13%  | [0.842, 0.887]          |
| Random Forest       | 85.98%  | ±0.78%  | [0.845, 0.875]          |
| Gradient Boosting   | 87.23%  | ±1.03%  | [0.852, 0.892]          |
| Voting Ensemble     | 87.13%  | ±0.99%  | [0.852, 0.891]          |

>  **Why CV scores are higher than Test scores?**
>
> This is expected and not a bug:
> - **CV scores** were measured on **resampled training data** (after SMOTETomek),
>   where churn rate is balanced at ~50% — making prediction easier.
> - **Test scores** were measured on the **original imbalanced data**,
>   where churn rate is only 26.5% — which is the real-world scenario.
>
> This gap is a sign of **honest evaluation**, not overfitting.
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
- Voting Ensemble
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
├── churn_model.pkl
├── scaler.pkl
├── selector.pkl
└── all_features.pkl
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
<img width="1918" height="953" alt="image" src="https://github.com/user-attachments/assets/b4d27a08-a97d-4d20-8642-77d3f6f3aea3" />

### SHAP Explainability
<img width="1350" height="2240" alt="127 0 0 1_5000_" src="https://github.com/user-attachments/assets/ccbb1ccd-8df1-4359-a841-0ca7fdfb06a9" />

### Confusion Matrix
<img width="1096" height="880" alt="confusion_matrix" src="https://github.com/user-attachments/assets/02d16366-fde8-4e6c-8ebe-b9308d41b23f" />

### Batch Prediction
<img width="1350" height="2399" alt="127 0 0 1_5000_ (1)" src="https://github.com/user-attachments/assets/ba022883-846e-46f0-b6b3-a56a7edec5ff" />

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

---

## Key Insights

- Customers with month-to-month contracts showed the highest churn probability.
- Fiber optic users had higher churn risk compared to DSL users.
- Long-term customers were significantly less likely to churn.
- Electronic check payment method was strongly associated with customer churn.

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
