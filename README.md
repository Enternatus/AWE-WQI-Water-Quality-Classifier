# 💧 Water Quality Intelligence (WQI) Classifier

A robust, leakage-aware machine learning framework for **multi-class water quality classification**, integrating feature selection, class imbalance handling, and a novel **Accuracy-Weighted Ensemble (AWE)** for high-fidelity environmental prediction.

---

## 🚀 Overview

This project implements an end-to-end supervised learning system that classifies water samples into **five standardized quality categories** derived from the Water Quality Index (WQI):

| Class       | WQI Range |
|------------|----------|
| Excellent  | 0 – 25   |
| Good       | 26 – 50  |
| Moderate   | 51 – 75  |
| Poor       | 76 – 100 |
| Very Poor  | > 100    |

The system combines:
- Statistical preprocessing  
- Embedded feature selection  
- Leakage-free resampling (SMOTE)  
- Stratified cross-validation  
- Multi-model ensemble learning  

---

## 🧠 Core Contribution: AWE-WQI Ensemble

The proposed **Adaptive Weighted Ensemble (AWE-WQI)** aggregates predictions from multiple classifiers using a performance-driven weighting scheme.

**Weighting:**
w_i = (F1_i^2) / Σ(F1_j^2)

**Final Prediction:**
ŷ = argmax Σ (w_i · P_i(y|x))



This approach amplifies high-performing models while suppressing weaker ones, improving robustness and generalization.

---

## ⚙️ Methodology Summary

The pipeline follows a structured sequence:

1. Data preprocessing (mean imputation)
2. WQI discretization into 5 classes
3. Train-test split (stratified, 80/20)
4. Feature selection (Random Forest, median threshold)
5. Feature standardization (Z-score)
6. SMOTE applied within cross-validation folds
7. Model training:
   - Random Forest  
   - Gradient Boosting  
   - AdaBoost  
   - XGBoost  
8. Hyperparameter tuning (Stratified K-Fold)
9. AWE ensemble aggregation
10. Evaluation using weighted metrics

---

## 📁 Project Structure

AWE-WQI-Water-Quality-Classifier/
│
├── water_ml.py # Complete ML pipeline (training, validation, ensemble)
├── Results_MADE.csv # Dataset (295 samples, 9 features + WQI)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data_preprocessing/ # Logical stage
│ ├── Missing value imputation (mean)
│ ├── WQI classification (5 classes)
│ └── Label encoding
│
├── feature_engineering/ # Logical stage
│ ├── Feature selection (Random Forest)
│ └── Feature scaling (StandardScaler)
│
├── validation/ # Logical stage
│ ├── K-Fold cross-validation
│ └── Stratified K-Fold
│
├── imbalance_handling/ # Logical stage
│ └── SMOTE (inside pipeline)
│
├── models/ # Logical stage
│ ├── Random Forest
│ ├── Gradient Boosting
│ ├── AdaBoost
│ └── XGBoost
│
├── tuning/ # Logical stage
│ └── Hyperparameter tuning
│
└── ensemble/
└── AWE-WQI (Accuracy Weighted Ensemble)

## 📂 Code & Data

This repository contains all components required for full reproducibility:

- `water_ml.py` → Complete ML pipeline  
- `Results_MADE.csv` → Dataset (295 samples, 9 features + WQI)  

---

## ▶️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt

Run the pipeline:

python water_ml.py

The script performs:

Model training
Cross-validation
Hyperparameter tuning
Ensemble evaluation

📊 Evaluation Metrics

Models are evaluated using:

Accuracy
Precision (weighted)
Recall (weighted)
F1 Score (weighted)

Stratified K-Fold is used to ensure balanced class representation during validation.

🧪 Model Stack
Model	Purpose
Random Forest	Feature selection + baseline
Gradient Boosting	Residual learning
AdaBoost	Adaptive reweighting
XGBoost	Regularized boosting

🔒 Reproducibility
Fixed random_state = 42
Deterministic preprocessing and splitting
SMOTE applied correctly within training folds


📄 Research Paper

The complete methodology, experimental results, and analysis are documented in the accompanying research paper.

If you use this work, cite:
Water Quality Intelligence Classifier using AWE Ensemble and Stratified Validation, 2026.


📜 License

MIT License
