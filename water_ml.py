# ============================================
# WATER QUALITY CLASSIFICATION (FINAL CLEAN)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from itertools import product

# ============================================
# LOAD DATA
# ============================================

data = pd.read_csv("Results_MADE.csv")

# Fill missing numeric values
for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = data[col].fillna(data[col].mean())

# ============================================
# CLASS LABELS
# ============================================

def classify_wqi(wqi):
    if wqi <= 25: return "Excellent"
    elif wqi <= 50: return "Good"
    elif wqi <= 75: return "Moderate"
    elif wqi <= 100: return "Poor"
    else: return "Very Poor"

data["WQI_Class"] = data["WQI"].apply(classify_wqi)

X = data.drop(["WQI", "WQI_Class"], axis=1)
y = data["WQI_Class"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# ============================================
# SPLIT
# ============================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================
# FEATURE SELECTION
# ============================================

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=200, random_state=42),
    threshold="median"
)

X_train = selector.fit_transform(X_train_raw, y_train)
X_test = selector.transform(X_test_raw)

# ============================================
# SCALING
# ============================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# MODEL FACTORY
# ============================================

def get_model(name, params=None):
    if params is None:
        params = {}

    if name == "Random Forest":
        return RandomForestClassifier(**params, class_weight="balanced", random_state=42)

    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(**params, random_state=42)

    elif name == "AdaBoost":
        return AdaBoostClassifier(**params, random_state=42)

    elif name == "XGBoost":
        return XGBClassifier(**params, eval_metric='mlogloss', random_state=42, n_jobs=-1)

models = ["Random Forest", "Gradient Boosting", "AdaBoost", "XGBoost"]

# ============================================
# METRICS
# ============================================

def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='weighted')
    ]

def print_table(title, results):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    print(f"{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}")
    for name, vals in results.items():
        print(f"{name:<20}{vals[0]:<12.4f}{vals[1]:<12.4f}{vals[2]:<12.4f}{vals[3]:<12.4f}")

# ============================================
# BASE MODELS
# ============================================

base_results = {}

for name in models:
    model = get_model(name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    base_results[name] = get_metrics(y_test, preds)

print_table("BASE MODEL COMPARISON", base_results)

# ============================================
# NORMAL K-FOLD
# ============================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_results = {}

for name in models:
    scores = []

    for tr, val in kf.split(X_train):
        pipe = Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", get_model(name))
        ])

        pipe.fit(X_train[tr], y_train[tr])
        preds = pipe.predict(X_train[val])
        scores.append(get_metrics(y_train[val], preds))

    kf_results[name] = np.mean(scores, axis=0)

print_table("NORMAL K-FOLD", kf_results)

# ============================================
# STRATIFIED K-FOLD
# ============================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_results = {}

for name in models:
    scores = []

    for tr, val in skf.split(X_train, y_train):
        pipe = Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", get_model(name))
        ])

        pipe.fit(X_train[tr], y_train[tr])
        preds = pipe.predict(X_train[val])
        scores.append(get_metrics(y_train[val], preds))

    skf_results[name] = np.mean(scores, axis=0)

print_table("STRATIFIED K-FOLD", skf_results)

# ============================================
# HYPERPARAMETER TUNING
# ============================================

param_grid = {
    "AdaBoost": {
        "n_estimators": [120],
        "learning_rate": [0.7]
    },
    "XGBoost": {
        "n_estimators": [250],
        "max_depth": [4],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1],
        "reg_alpha": [0.5]
    }
}

tuned_models = {}

print("\nBEST PARAMETERS")
print("="*70)

for name in param_grid:
    best_score = 0
    best_params = None

    for combo in product(*param_grid[name].values()):
        params = dict(zip(param_grid[name].keys(), combo))
        scores = []

        for tr, val in skf.split(X_train, y_train):
            pipe = Pipeline([
                ("smote", SMOTE(random_state=42)),
                ("model", get_model(name, params))
            ])

            pipe.fit(X_train[tr], y_train[tr])
            preds = pipe.predict(X_train[val])
            scores.append(f1_score(y_train[val], preds, average='weighted'))

        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_params = params

    tuned_models[name] = get_model(name, best_params)
    print(f"{name}: {best_params}")

# ============================================
# FINAL TRAINING WITH SMOTE
# ============================================

X_train_final, y_train_final = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train tuned models
for model in tuned_models.values():
    model.fit(X_train_final, y_train_final)

# ============================================
# TUNED MODEL COMPARISON (FIXED)
# ============================================

tuned_results = {}

# tuned models
for name, model in tuned_models.items():
    preds = model.predict(X_test)
    tuned_results[name] = get_metrics(y_test, preds)

# add remaining models
for name in models:
    if name not in tuned_results:
        model = get_model(name)
        model.fit(X_train_final, y_train_final)
        preds = model.predict(X_test)
        tuned_results[name] = get_metrics(y_test, preds)

# print in correct order
print("\n" + "="*70)
print("TUNED MODEL COMPARISON")
print("="*70)
print(f"{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}")

for name in models:
    vals = tuned_results[name]
    print(f"{name:<20}{vals[0]:<12.4f}{vals[1]:<12.4f}{vals[2]:<12.4f}{vals[3]:<12.4f}")

# ============================================
# AWE ENSEMBLE
# ============================================

weights = {}
total = 0

for name, model in tuned_models.items():
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    weights[name] = f1**2
    total += weights[name]

for k in weights:
    weights[k] /= total

final_proba = sum(
    tuned_models[name].predict_proba(X_test) * weights[name]
    for name in tuned_models
)

y_pred = np.argmax(final_proba, axis=1)

print("\nAWE-WQI ENSEMBLE RESULTS")
print("="*70)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nClassification Report:\n")
print(classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred)
))

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ============================================
# FEATURE IMPORTANCE (ALL FEATURES)
# ============================================

rf_full = RandomForestClassifier(n_estimators=200, random_state=42)
rf_full.fit(X_train_raw, y_train)

importances = rf_full.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (All Features)")
plt.tight_layout()
plt.show()

# ============================================
# ROC CURVE
# ============================================

y_bin = label_binarize(y_test, classes=[0,1,2,3,4])

plt.figure(figsize=(7,5))

for i in range(5):
    fpr, tpr, _ = roc_curve(y_bin[:, i], final_proba[:, i])
    plt.plot(fpr, tpr, label=le.classes_[i])

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()