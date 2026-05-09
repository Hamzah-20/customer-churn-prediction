import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import warnings
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

app = Flask(__name__)

print("=" * 60)
print("Loading model components...")
print("=" * 60)


# ============================================================================
#  Definition of FeatureSelector
# ============================================================================
class FeatureSelector:

    def __init__(self, features):
        self.features = features

    def transform(self, X):
        return X[self.features]

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return self.features


# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

try:
    with open('full_pipeline.pkl', 'rb') as f:
        full_pipeline = pickle.load(f)
    print(f"✅ Loaded full_pipeline.pkl")
except FileNotFoundError:
    print("⚠️ full_pipeline.pkl not found — will use legacy files")
    full_pipeline = None

try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Loaded churn_model.pkl")
except Exception as e:
    print(f"❌ {e}");
    model = None

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Loaded scaler.pkl")
except Exception as e:
    print(f"❌ {e}");
    scaler = None

try:
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    print(f"✅ Loaded selected_features.pkl: {len(selected_features)} features")
except Exception as e:
    print(f"❌ {e}");
    selected_features = []

try:
    with open('all_feature_names.pkl', 'rb') as f:
        all_feature_names = pickle.load(f)
    print(f"✅ Loaded all_feature_names.pkl: {len(all_feature_names)} features")
except FileNotFoundError:
    try:
        with open('full_features.pkl', 'rb') as f:
            all_feature_names = pickle.load(f)
        print(f"✅ Loaded full_features.pkl: {len(all_feature_names)} features")
    except Exception as e:
        print(f"❌ {e}");
        all_feature_names = []

try:
    with open('encoding_info.pkl', 'rb') as f:
        encoding_info = pickle.load(f)
    print("✅ Loaded encoding_info.pkl")
except FileNotFoundError:
    encoding_info = None
    print("⚠️ encoding_info.pkl not found")

print("=" * 60)
print(f"Pipeline ready | Selected features: {len(selected_features)}")
print("=" * 60)


# ============================================================================
# HELPER: FEATURE ENGINEERING
# ============================================================================
def engineer_features(df_in):

    df_out = df_in.copy()

    df_out['is_new_customer'] = (df_out['tenure'] < 12).astype(int)
    df_out['is_very_new'] = (df_out['tenure'] < 3).astype(int)
    df_out['is_long_term'] = (df_out['tenure'] > 60).astype(int)
    df_out['tenure_years'] = df_out['tenure'] / 12
    df_out['tenure_squared'] = df_out['tenure'] ** 2

    df_out['avg_monthly_charge'] = df_out['TotalCharges'] / (df_out['tenure'] + 1)
    df_out['charge_vs_avg'] = df_out['MonthlyCharges'] - df_out['avg_monthly_charge']
    df_out['high_charger'] = (df_out['MonthlyCharges'] > 65).astype(int)
    df_out['charge_ratio'] = df_out['MonthlyCharges'] / (df_out['avg_monthly_charge'] + 0.01)

    all_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for s in all_services:
        if s in df_out.columns:
            df_out[f'no_{s}'] = (df_out[s] == 'No').astype(int)
            df_out[f'yes_{s}'] = (df_out[s] == 'Yes').astype(int)

    no_cols = [f'no_{s}' for s in all_services if f'no_{s}' in df_out.columns]
    df_out['total_missing_services'] = df_out[no_cols].sum(axis=1) if no_cols else 0

    df_out['high_risk_customer'] = ((df_out['tenure'] < 12) & (df_out['MonthlyCharges'] > 70)).astype(int)
    df_out['low_tenure_high_charge'] = ((df_out['tenure'] < 6) & (df_out['MonthlyCharges'] > 80)).astype(int)

    if 'Contract' in df_out.columns:
        df_out['is_monthly_contract'] = (df_out['Contract'] == 'Month-to-month').astype(int)
        df_out['is_yearly_contract'] = (df_out['Contract'] == 'One year').astype(int)
        df_out['is_two_year_contract'] = (df_out['Contract'] == 'Two year').astype(int)

    if 'PaymentMethod' in df_out.columns:
        df_out['is_electronic_check'] = (df_out['PaymentMethod'] == 'Electronic check').astype(int)

    return df_out


def align_to_training(df_encoded):

    for col in all_feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[all_feature_names]


def predict_with_pipeline(X_df):

    if full_pipeline is not None:
        prob = full_pipeline.predict_proba(X_df)[0][1]
        pred = full_pipeline.predict(X_df)[0]
    else:
        X_scaled = scaler.transform(X_df.values)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = model.predict(X_scaled)[0]
    return float(prob), int(pred)


def predict_batch_with_pipeline(X_df):

    if full_pipeline is not None:
        probs = full_pipeline.predict_proba(X_df)[:, 1]
        preds = full_pipeline.predict(X_df)
    else:
        X_scaled = scaler.transform(X_df.values)
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = model.predict(X_scaled)
    return probs, preds


def get_risk_info(probability):
    p = probability * 100
    if p > 70:
        return "Very High 🔴", "Contact customer immediately! Offer 30% retention discount and free TechSupport for 3 months"
    elif p > 50:
        return "High 🟠", "Contact customer soon. Send 15% discount on next bill"
    elif p > 30:
        return "Medium 🟡", "Send promotional email. Offer free additional service"
    else:
        return "Low 🟢", "Customer is safe. Track satisfaction through monthly survey"


# ============================================================================
# FORM FIELDS
# ============================================================================
def get_dynamic_form_fields():
    original_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'MonthlyCharges', 'TotalCharges', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    default_options = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': ['0', '1'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check',
                          'Bank transfer (automatic)', 'Credit card (automatic)']
    }

    try:
        df_orig = pd.read_csv('Telco-Customer-Churn.csv')
        for col in original_columns:
            if col in df_orig.columns:
                vals = df_orig[col].dropna().unique().tolist()
                vals = [str(v).strip() for v in vals if str(v) != 'nan']
                if vals:
                    default_options[col] = vals
    except Exception:
        pass

    fields = []
    for col in original_columns:
        if col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
            ftype = 'number'
        else:
            ftype = 'select'

        label = col.replace('_', ' ').replace('Charges', 'Charges ($)').title()
        field = {'name': col, 'label': label, 'type': ftype}

        if ftype == 'select':
            field['options'] = default_options.get(col, ['Yes', 'No'])

        fields.append(field)

    return fields


# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/form_fields')
def form_fields():
    return jsonify({'success': True, 'fields': get_dynamic_form_fields()})


@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        data = request.json
        print(f"\n{'=' * 50}")
        print(f"📊 Single prediction request received")

        row = {
            'tenure': float(data.get('tenure', 0)),
            'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
            'TotalCharges': float(data.get('TotalCharges', 0)),
            'SeniorCitizen': int(float(data.get('SeniorCitizen', 0))),
            'gender': str(data.get('gender', 'Male')),
            'Partner': str(data.get('Partner', 'No')),
            'Dependents': str(data.get('Dependents', 'No')),
            'PhoneService': str(data.get('PhoneService', 'Yes')),
            'MultipleLines': str(data.get('MultipleLines', 'No')),
            'InternetService': str(data.get('InternetService', 'DSL')),
            'OnlineSecurity': str(data.get('OnlineSecurity', 'No')),
            'OnlineBackup': str(data.get('OnlineBackup', 'No')),
            'DeviceProtection': str(data.get('DeviceProtection', 'No')),
            'TechSupport': str(data.get('TechSupport', 'No')),
            'StreamingTV': str(data.get('StreamingTV', 'No')),
            'StreamingMovies': str(data.get('StreamingMovies', 'No')),
            'Contract': str(data.get('Contract', 'Month-to-month')),
            'PaperlessBilling': str(data.get('PaperlessBilling', 'Yes')),
            'PaymentMethod': str(data.get('PaymentMethod', 'Electronic check')),
        }

        if row['TotalCharges'] == 0 and row['tenure'] > 0:
            row['TotalCharges'] = row['MonthlyCharges'] * row['tenure']

        print(f"   Input: tenure={row['tenure']}, monthly={row['MonthlyCharges']}, "
              f"contract={row['Contract']}, internet={row['InternetService']}")

        df_row = pd.DataFrame([row])

        df_row = engineer_features(df_row)

        cat_cols = df_row.select_dtypes(include=['object']).columns.tolist()
        df_encoded = pd.get_dummies(df_row, columns=cat_cols, drop_first=True)

        df_aligned = align_to_training(df_encoded)
        print(f"   After alignment: {df_aligned.shape[1]} features")

        missing_sel = [f for f in selected_features if f not in df_aligned.columns]
        if missing_sel:
            print(f"   ⚠️ Missing in selected: {missing_sel}")
            for f in missing_sel:
                df_aligned[f] = 0

        df_final = df_aligned[selected_features]
        print(f"   After selection: {df_final.shape[1]} features")

        probability, prediction = predict_with_pipeline(df_final)
        print(f"   ✅ Probability: {probability * 100:.1f}% | Prediction: {'Churn' if prediction else 'Stay'}")

        risk, recommendation = get_risk_info(probability)

        return jsonify({
            'success': True,
            'churn_probability': round(probability * 100, 1),
            'prediction': 'Will Churn' if prediction == 1 else 'Will Stay',
            'risk_level': risk,
            'recommendation': recommendation
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/upload_batch', methods=['POST'])
def predict_batch():

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'})
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a CSV file only'})

        df = pd.read_csv(file)
        print(f"\n{'=' * 50}")
        print(f"📂 Batch: {len(df)} rows, {len(df.columns)} cols")

        customer_ids = (df['customerID'].tolist()
                        if 'customerID' in df.columns
                        else [f'Customer {i + 1}' for i in range(len(df))])

        for col in ['Churn', 'customerID']:
            if col in df.columns:
                df = df.drop(col, axis=1)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        print(f"   After cleaning: {len(df)} rows")

        mask = (df['TotalCharges'] == 0) & (df['tenure'] > 0)
        df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']

        df = engineer_features(df)

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"   After encoding: {df_encoded.shape[1]} features")

        df_aligned = align_to_training(df_encoded)
        print(f"   After alignment: {df_aligned.shape[1]} features")

        missing_sel = [f for f in selected_features if f not in df_aligned.columns]
        if missing_sel:
            print(f"   ⚠️ Missing in selected: {missing_sel}")
            for f in missing_sel:
                df_aligned[f] = 0

        df_final = df_aligned[selected_features]
        print(f"   After selection: {df_final.shape[1]} features")

        expected_features = len(selected_features)
        actual_features = df_final.shape[1]
        print(f"   Expected: {expected_features} | Got: {actual_features}")

        if actual_features != expected_features:
            return jsonify({
                'success': False,
                'error': f'Feature mismatch: expected {expected_features}, got {actual_features}. '
                         f'Please re-run model.py'
            })

        probabilities, predictions = predict_batch_with_pipeline(df_final)
        print(f"   ✅ Predictions complete: {len(predictions)} customers")

        results_df = pd.DataFrame({
            'Customer_ID': customer_ids[:len(predictions)],
            'Prediction': ['Will Churn' if p == 1 else 'Will Stay' for p in predictions],
            'Churn_Probability': [round(p * 100, 1) for p in probabilities],
            'Risk_Level': ['High' if p > 0.7
                           else 'Medium' if p > 0.4
            else 'Low' for p in probabilities]
        })

        output_path = 'static/predictions_result.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        plt.figure(figsize=(14, 6))
        colors = ['#2ecc71' if x == 'Will Stay' else '#e74c3c'
                  for x in results_df['Prediction']]
        plt.bar(range(len(results_df)), results_df['Churn_Probability'],
                color=colors, alpha=0.7)
        plt.axhline(y=50, color='orange', linestyle='--', linewidth=2,
                    label='Risk Threshold (50%)')
        plt.xlabel('Customer Number', fontsize=12)
        plt.ylabel('Churn Probability (%)', fontsize=12)
        plt.title('Customer Churn Prediction Results — Batch Analysis', fontsize=14)
        plt.legend();
        plt.ylim(0, 100);
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/batch_chart.png', dpi=150)
        plt.close()

        churn_count = int((results_df['Prediction'] == 'Will Churn').sum())

        return jsonify({
            'success': True,
            'total_customers': len(results_df),
            'churn_count': churn_count,
            'churn_percentage': round(churn_count / len(results_df) * 100, 1),
            'stay_count': len(results_df) - churn_count,
            'download_url': '/static/predictions_result.csv',
            'chart_url': '/static/batch_chart.png?v=' + str(
                os.path.getmtime('static/batch_chart.png'))
            if os.path.exists('static/batch_chart.png') else None,
            'results': results_df.head(30).to_dict('records')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/download_results')
def download_results():
    return send_file('static/predictions_result.csv',
                     as_attachment=True, download_name='churn_predictions.csv')


@app.route('/insights')
def get_insights():
    images = {}
    image_files = ['shap_summary', 'shap_importance', 'tenure_effect',
                   'price_effect', 'contract_effect', 'services_effect', 'confusion_matrix']
    for img in image_files:
        path = f'static/{img}.png'
        images[img] = f'/static/{img}.png' if os.path.exists(path) else None
    return jsonify(images)


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    print("\n" + "=" * 60)
    print("🚀 Starting Churn Prediction Application")
    print("=" * 60)
    print(f"   Pipeline:         {'✅ full_pipeline.pkl' if full_pipeline else '⚠️ legacy files'}")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   All features:      {len(all_feature_names)}")
    print(f"   URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)