import os
import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file


import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

app = Flask(__name__)


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

PIPELINE_PATH = MODELS_DIR / "full_pipeline.pkl"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.pkl"
ALL_FEATURE_NAMES_PATH = MODELS_DIR / "all_feature_names.pkl"
ENCODING_INFO_PATH = MODELS_DIR / "encoding_info.pkl"

DATASET_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
BATCH_RESULTS_PATH = STATIC_DIR / "predictions_result.csv"
BATCH_CHART_PATH = STATIC_DIR / "batch_chart.png"

RISK_THRESHOLD = 0.65


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

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

def load_pickle_file(file_path: Path, description: str):
    """Load a required pickle artifact with a clear error message."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found at: {file_path}"
        )

    try:
        with file_path.open("rb") as file:
            artifact = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise RuntimeError(
            f"Failed to load {description} from {file_path}: {error}"
        ) from error

    print(f"✅ Loaded {description}")
    return artifact


print("=" * 60)
print("Loading model components...")
print("=" * 60)

try:
    full_pipeline = load_pickle_file(
        PIPELINE_PATH,
        "full_pipeline.pkl",
    )

    selected_features = load_pickle_file(
        SELECTED_FEATURES_PATH,
        "selected_features.pkl",
    )

    all_feature_names = load_pickle_file(
        ALL_FEATURE_NAMES_PATH,
        "all_feature_names.pkl",
    )

    encoding_info = load_pickle_file(
        ENCODING_INFO_PATH,
        "encoding_info.pkl",
    )

except (FileNotFoundError, RuntimeError) as error:
    print(f"❌ Model initialization failed: {error}")
    raise SystemExit(1) from error


if not selected_features:
    raise SystemExit(
        "❌ selected_features.pkl does not contain any features."
    )

if not all_feature_names:
    raise SystemExit(
        "❌ all_feature_names.pkl does not contain any features."
    )


print(
    f"✅ Selected features: {len(selected_features)}"
)

print(
    f"✅ All training features: {len(all_feature_names)}"
)

print("=" * 60)
print("Production pipeline is ready.")
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



def predict_with_pipeline(
    input_data: pd.DataFrame,
) -> tuple[float, int]:
    """Predict churn for one customer."""

    probability = float(
        full_pipeline.predict_proba(input_data)[0][1]
    )

    prediction = int(
        probability >= RISK_THRESHOLD
    )

    return probability, prediction


def predict_batch_with_pipeline(
    input_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict churn for multiple customers."""

    probabilities = full_pipeline.predict_proba(
        input_data
    )[:, 1]

    predictions = (
        probabilities >= RISK_THRESHOLD
    ).astype(int)

    return probabilities, predictions




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
        df_orig = pd.read_csv(DATASET_PATH)
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

        customer_ids = (
            df['customerID'].tolist()
            if 'customerID' in df.columns
            else [f'Customer {i + 1}' for i in range(len(df))]
        )

        for col in ['Churn', 'customerID']:
            if col in df.columns:
                df = df.drop(col, axis=1)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)

        print(f"   After cleaning: {len(df)} rows")

        customer_ids = customer_ids[:len(df)]

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
                'error': f'Feature mismatch: expected {expected_features}, got {actual_features}. Please re-run model.py'
            })

        probabilities, predictions = predict_batch_with_pipeline(df_final)


        print(f"   ✅ Predictions complete: {len(predictions)} customers")
        print(f"   Threshold used: {RISK_THRESHOLD * 100:.0f}%")
        print(f"   Probability mean: {probabilities.mean() * 100:.2f}%")
        print(f"   Probability min: {probabilities.min() * 100:.2f}%")
        print(f"   Probability max: {probabilities.max() * 100:.2f}%")

        results_df = pd.DataFrame({
            'Customer_ID': customer_ids[:len(predictions)],
            'Prediction': ['Will Churn' if p == 1 else 'Will Stay' for p in predictions],
            'Churn_Probability': [round(p * 100, 1) for p in probabilities],
            "Risk_Level": [
                "High"
                if probability >= RISK_THRESHOLD
                else "Medium"
                if probability >= 0.40
                else "Low"
                for probability in probabilities
            ]
        })

        results_df.to_csv(BATCH_RESULTS_PATH,index=False, encoding="utf-8-sig",)

        plt.figure(figsize=(14, 6))

        colors = [
            '#e74c3c' if x == 'Will Churn' else '#2ecc71'
            for x in results_df['Prediction']
        ]

        plt.bar(
            range(len(results_df)),
            results_df['Churn_Probability'],
            color=colors,
            alpha=0.7
        )

        plt.axhline(
            y=RISK_THRESHOLD * 100,
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Risk Threshold ({int(RISK_THRESHOLD * 100)}%)'
        )

        plt.xlabel('Customer Number', fontsize=12)
        plt.ylabel('Churn Probability (%)', fontsize=12)
        plt.title('Customer Churn Prediction Results — Batch Analysis', fontsize=14)
        plt.legend()
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(BATCH_CHART_PATH,dpi=150, bbox_inches="tight",)
        plt.close()

        churn_count = int((results_df['Prediction'] == 'Will Churn').sum())

        return jsonify({
            'success': True,
            'total_customers': len(results_df),
            'churn_count': churn_count,
            'churn_percentage': round(churn_count / len(results_df) * 100, 1),
            'stay_count': len(results_df) - churn_count,
            'download_url': '/static/predictions_result.csv',
            "chart_url": (
                "/static/batch_chart.png?v="
                + str(BATCH_CHART_PATH.stat().st_mtime)
                if BATCH_CHART_PATH.exists()
                else None
            ),
            'results': results_df.head(30).to_dict('records')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route("/download_results")
def download_results():
    if not BATCH_RESULTS_PATH.exists():
        return jsonify(
            {
                "success": False,
                "error": "No batch prediction results are available.",
            }
        ), 404

    return send_file(
        BATCH_RESULTS_PATH,
        as_attachment=True,
        download_name="churn_predictions.csv",
    )


@app.route('/insights')
def get_insights():
    images = {}
    image_files = ['shap_summary', 'shap_importance', 'tenure_effect',
                   'price_effect', 'contract_effect', 'services_effect', 'confusion_matrix']
    for img in image_files:
        path = f'static/{img}.png'
        images[img] = f'/static/{img}.png' if os.path.exists(path) else None
    return jsonify(images)


if __name__ == "__main__":
    STATIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMPLATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("🚀 Starting Churn Prediction Application")
    print("=" * 60)
    print("   Pipeline: Production ML Pipeline")
    print(f"   Risk threshold: {RISK_THRESHOLD:.2f}")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   All features: {len(all_feature_names)}")
    print("   Local URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
