import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score, confusion_matrix)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_rel
import shap
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CUSTOMER CHURN PREDICTION - PROFESSIONAL EDITION")
print("=" * 80)

# ============================================================================
# 1. LOADING DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

print(f"   Loaded {len(df):,} customers")
print(f"   Churn rate: {df['Churn'].mean() * 100:.1f}%")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n2. Creating advanced features...")

def engineer_features(df_in):
    """
    دالة مركزية لعمل الـ feature engineering
    بتُستخدم في التدريب والـ app بنفس الطريقة
    """
    df_out = df_in.copy()

    df_out['is_new_customer']        = (df_out['tenure'] < 12).astype(int)
    df_out['is_very_new']            = (df_out['tenure'] < 3).astype(int)
    df_out['is_long_term']           = (df_out['tenure'] > 60).astype(int)
    df_out['tenure_years']           = df_out['tenure'] / 12
    df_out['tenure_squared']         = df_out['tenure'] ** 2

    df_out['avg_monthly_charge']     = df_out['TotalCharges'] / (df_out['tenure'] + 1)
    df_out['charge_vs_avg']          = df_out['MonthlyCharges'] - df_out['avg_monthly_charge']
    df_out['high_charger']           = (df_out['MonthlyCharges'] > df_out['MonthlyCharges'].median()).astype(int)
    df_out['charge_ratio']           = df_out['MonthlyCharges'] / (df_out['avg_monthly_charge'] + 0.01)

    all_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for s in all_services:
        if s in df_out.columns:
            df_out[f'no_{s}']  = (df_out[s] == 'No').astype(int)
            df_out[f'yes_{s}'] = (df_out[s] == 'Yes').astype(int)

    no_cols = [f'no_{s}' for s in all_services if f'no_{s}' in df_out.columns]
    df_out['total_missing_services'] = df_out[no_cols].sum(axis=1) if no_cols else 0

    df_out['high_risk_customer']     = ((df_out['tenure'] < 12) & (df_out['MonthlyCharges'] > 70)).astype(int)
    df_out['low_tenure_high_charge'] = ((df_out['tenure'] < 6)  & (df_out['MonthlyCharges'] > 80)).astype(int)

    if 'Contract' in df_out.columns:
        df_out['is_monthly_contract']  = (df_out['Contract'] == 'Month-to-month').astype(int)
        df_out['is_yearly_contract']   = (df_out['Contract'] == 'One year').astype(int)
        df_out['is_two_year_contract'] = (df_out['Contract'] == 'Two year').astype(int)

    if 'PaymentMethod' in df_out.columns:
        df_out['is_electronic_check']  = (df_out['PaymentMethod'] == 'Electronic check').astype(int)

    return df_out


df = engineer_features(df)
print(f"   Created {len(df.columns)} total features")

# ============================================================================
# 3. ENCODING
# ============================================================================
print("\n3. Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(f"   Categorical variables: {categorical_cols}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"   After encoding: {df_encoded.shape[1]} features")

# ============================================================================
# 4. SPLIT
# ============================================================================
print("\n4. Splitting data...")
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

all_feature_names = X.columns.tolist()
print(f"   Total features: {len(all_feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {X_train.shape[0]:,} | Testing: {X_test.shape[0]:,}")

# ============================================================================
# 5. CLASS IMBALANCE
# ============================================================================
print("\n5. Handling class imbalance...")
smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
print(f"   Before: {len(y_train):,} → After: {len(y_train_res):,}")
print(f"   Churn rate after: {y_train_res.mean() * 100:.1f}%")

# ============================================================================
# 6. FEATURE SELECTION
# ============================================================================
print("\n6. Feature Selection...")

k_features = min(45, X_train_res.shape[1])
selector_kbest = SelectKBest(score_func=mutual_info_classif, k=k_features)
selector_kbest.fit(X_train_res, y_train_res)
selected_kbest = X.columns[selector_kbest.get_support()].tolist()
print(f"   KBest selected: {len(selected_kbest)} features")

l1_selector = SelectFromModel(
    LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
)
l1_selector.fit(X_train_res, y_train_res)
selected_l1 = X.columns[l1_selector.get_support()].tolist()
print(f"   L1 selected: {len(selected_l1)} features")

selected_features = list(set(selected_kbest) & set(selected_l1))
if len(selected_features) < 20:
    selected_features = selected_kbest
    print(f"   Using KBest (intersection too small)")
else:
    print(f"   Final (KBest ∩ L1): {len(selected_features)} features")

X_train_sel = X_train_res[selected_features]
X_test_sel  = X_test[selected_features]

print(f"\n   Selected features list:")
for i, f in enumerate(sorted(selected_features)):
    print(f"     {i+1:2d}. {f}")

# ============================================================================
# 7. PIPELINES
# ============================================================================
print("\n7. Creating ML Pipelines...")

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, max_features='sqrt',
        class_weight='balanced', random_state=42
    ))
])

gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        min_samples_split=20, min_samples_leaf=10,
        subsample=0.8, random_state=42
    ))
])

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        C=1.0, class_weight='balanced', random_state=42, max_iter=1000
    ))
])

# ============================================================================
# 8. TRAINING
# ============================================================================
print("\n8. Training models...")

def evaluate_pipeline(pipeline, Xtr, ytr, Xte, yte, name):
    pipeline.fit(Xtr, ytr)
    yp    = pipeline.predict(Xte)
    yprob = pipeline.predict_proba(Xte)[:, 1]
    return {
        'name': name,
        'accuracy':  accuracy_score(yte, yp),
        'precision': precision_score(yte, yp),
        'recall':    recall_score(yte, yp),
        'f1':        f1_score(yte, yp),
        'auc':       roc_auc_score(yte, yprob),
        'predictions':   yp,
        'probabilities': yprob,
        'pipeline':  pipeline
    }

results = []
results.append(evaluate_pipeline(lr_pipeline, X_train_sel, y_train_res, X_test_sel, y_test, "Logistic Regression"))
results.append(evaluate_pipeline(rf_pipeline, X_train_sel, y_train_res, X_test_sel, y_test, "Random Forest"))
results.append(evaluate_pipeline(gb_pipeline, X_train_sel, y_train_res, X_test_sel, y_test, "Gradient Boosting"))

voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_pipeline.named_steps['classifier']),
        ('gb', gb_pipeline.named_steps['classifier'])
    ],
    voting='soft'
)
voting_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', voting_clf)
])
voting_pipeline.fit(X_train_sel, y_train_res)
vp    = voting_pipeline.predict(X_test_sel)
vprob = voting_pipeline.predict_proba(X_test_sel)[:, 1]
results.append({
    'name': 'Voting Ensemble',
    'accuracy':  accuracy_score(y_test, vp),
    'precision': precision_score(y_test, vp),
    'recall':    recall_score(y_test, vp),
    'f1':        f1_score(y_test, vp),
    'auc':       roc_auc_score(y_test, vprob),
    'predictions':   vp,
    'probabilities': vprob,
    'pipeline':  voting_pipeline
})

print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"))
print("-" * 70)
for r in results:
    print("{:<25} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
        r['name'], r['accuracy']*100, r['precision']*100,
        r['recall']*100, r['f1']*100, r['auc']*100))

# ============================================================================
# 9. CROSS-VALIDATION
# ============================================================================
print("\n9. Cross-Validation (5-Fold × 3 Repeats)...")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_results = {}

for r in results:
    scores = cross_val_score(r['pipeline'], X_train_sel, y_train_res,
                             cv=cv, scoring='f1', n_jobs=-1)
    cv_results[r['name']] = scores
    ci_lo = scores.mean() - 1.96 * scores.std()
    ci_hi = scores.mean() + 1.96 * scores.std()
    print(f"\n   {r['name']}:")
    print(f"     Mean F1: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    print(f"     95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

# ============================================================================
# 10. BEST MODEL
# ============================================================================
best_result = max(results, key=lambda x: x['f1'])
best_name   = best_result['name']
best_pipeline = best_result['pipeline']

print(f"\n🏆 Best Model: {best_name}")
print(f"   F1={best_result['f1']*100:.2f}% | Recall={best_result['recall']*100:.2f}% | AUC={best_result['auc']*100:.2f}%")

# ============================================================================
# 11. SAVE — الحل الجذري: نحفظ كل شيء بشكل صحيح
# ============================================================================
print("\n11. Saving model artifacts...")
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

with open('full_pipeline.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
print("   ✅ Saved full_pipeline.pkl (scaler + classifier)")

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
print(f"   ✅ Saved selected_features.pkl ({len(selected_features)} features)")

with open('all_feature_names.pkl', 'wb') as f:
    pickle.dump(all_feature_names, f)
print(f"   ✅ Saved all_feature_names.pkl ({len(all_feature_names)} features)")

encoding_info = {
    'categorical_cols': categorical_cols,
    'selected_features': selected_features,
    'all_feature_names': all_feature_names,
    'drop_first': True
}
with open('encoding_info.pkl', 'wb') as f:
    pickle.dump(encoding_info, f)
print("   ✅ Saved encoding_info.pkl")

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(best_pipeline.named_steps['classifier'], f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(best_pipeline.named_steps['scaler'], f)
with open('full_features.pkl', 'wb') as f:
    pickle.dump(all_feature_names, f)
with open('selector.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
print("   ✅ Saved legacy pkl files for backward compatibility")

# ============================================================================
# 12. VISUALIZATIONS
# ============================================================================
print("\n12. Creating visualizations...")

y_final_pred  = best_result['predictions']
y_final_proba = best_result['probabilities']

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
plt.title(f'Confusion Matrix — {best_name}')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ confusion_matrix.png")

df_orig = pd.read_csv('Telco-Customer-Churn.csv')
df_orig['TotalCharges'] = pd.to_numeric(df_orig['TotalCharges'], errors='coerce')
df_orig.dropna(inplace=True)
df_orig['Churn'] = df_orig['Churn'].map({'Yes': 1, 'No': 0})

plt.figure(figsize=(10, 6))
tenure_effect = df_orig.groupby('tenure')['Churn'].mean()
plt.plot(tenure_effect.index, tenure_effect.values, 'b-o', linewidth=2, markersize=4)
plt.axhline(y=df_orig['Churn'].mean(), color='r', linestyle='--',
            label=f'Overall avg ({df_orig["Churn"].mean()*100:.1f}%)')
plt.xlabel('Tenure (months)'); plt.ylabel('Churn Rate')
plt.title('Tenure vs Churn Rate')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('static/tenure_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ tenure_effect.png")

plt.figure(figsize=(12, 6))
price_bins = pd.cut(df_orig['MonthlyCharges'], bins=8)
price_effect = df_orig.groupby(price_bins, observed=True)['Churn'].mean()
price_effect.plot(kind='bar', color='coral', edgecolor='black')
plt.axhline(y=df_orig['Churn'].mean(), color='blue', linestyle='--',
            label=f'Average ({df_orig["Churn"].mean()*100:.1f}%)')
plt.xlabel('Monthly Charges ($)'); plt.ylabel('Churn Rate')
plt.title('Monthly Charges vs Churn Rate')
plt.xticks(rotation=45); plt.legend(); plt.tight_layout()
plt.savefig('static/price_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ price_effect.png")

if 'Contract' in df_orig.columns:
    plt.figure(figsize=(8, 6))
    contract_effect = df_orig.groupby('Contract')['Churn'].mean()
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    contract_effect.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Contract Type vs Churn Rate')
    plt.ylabel('Churn Rate'); plt.xticks(rotation=0)
    for i, v in enumerate(contract_effect):
        plt.text(i, v + 0.01, f'{v*100:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/contract_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ contract_effect.png")

services_list = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
service_effect = {}
for s in services_list:
    if s in df_orig.columns:
        no_val  = df_orig[df_orig[s] == 'No']['Churn'].mean()
        yes_val = df_orig[df_orig[s] == 'Yes']['Churn'].mean()
        service_effect[s] = [no_val, yes_val]

if service_effect:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(service_effect))
    w = 0.35
    no_vals  = [service_effect[s][0] for s in service_effect]
    yes_vals = [service_effect[s][1] for s in service_effect]
    plt.bar(x - w/2, no_vals,  w, label='Without Service', color='salmon')
    plt.bar(x + w/2, yes_vals, w, label='With Service',    color='lightgreen')
    plt.xlabel('Services'); plt.ylabel('Churn Rate')
    plt.title('Impact of Services on Churn')
    plt.xticks(x, list(service_effect.keys()), rotation=45)
    plt.legend(); plt.tight_layout()
    plt.savefig('static/services_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ services_effect.png")

# ============================================================================
# 13. SHAP ANALYSIS
# ============================================================================
print("\n13. SHAP Analysis...")
try:
    clf     = best_pipeline.named_steps['classifier']
    scaler_ = best_pipeline.named_steps['scaler']

    X_test_scaled = scaler_.transform(X_test_sel)
    X_sample = X_test_scaled[:min(200, len(X_test_scaled))]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    plt.figure(figsize=(14, 10))
    shap.summary_plot(sv, X_sample, feature_names=selected_features, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('static/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_sample, feature_names=selected_features,
                      show=False, plot_type="bar", max_display=15)
    plt.tight_layout()
    plt.savefig('static/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ shap_summary.png + shap_importance.png")
except Exception as e:
    print(f"   ⚠️ SHAP error: {e}")

# ============================================================================
# 14. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print(f"""
Best Model:      {best_name}
Accuracy:        {best_result['accuracy']*100:.2f}%
Precision:       {best_result['precision']*100:.2f}%
Recall:          {best_result['recall']*100:.2f}%
F1-Score:        {best_result['f1']*100:.2f}%
ROC-AUC:         {best_result['auc']*100:.2f}%

CV Mean F1:      {cv_results[best_name].mean()*100:.2f}% ± {cv_results[best_name].std()*100:.2f}%

Features:        {len(all_feature_names)} total → {len(selected_features)} selected

Key Files Saved:
  ✅ full_pipeline.pkl        ← الأهم: pipeline كامل (scaler + model)
  ✅ selected_features.pkl    ← الـ features المختارة بالترتيب
  ✅ all_feature_names.pkl    ← كل الـ features بعد الـ encoding
  ✅ encoding_info.pkl        ← معلومات الـ encoding
""")
print("=" * 80)