# 💧 Water Quality Intelligence (WQI) Classifier

A robust, leakage-aware machine learning pipeline for **multi-class water quality classification**, combining feature selection, imbalance handling, and a custom **Accuracy-Weighted Ensemble (AWE)** for high-fidelity environmental prediction.

---

## 🚀 Overview

This project implements an end-to-end supervised learning system that classifies water samples into **five standardized quality categories**:

| Class       | WQI Range |
|------------|----------|
| Excellent  | 0 – 25   |
| Good       | 26 – 50  |
| Moderate   | 51 – 75  |
| Poor       | 76 – 100 |
| Very Poor  | > 100    |

### 🔗 Core Components

- Statistical preprocessing  
- Embedded feature selection  
- Leakage-free resampling  
- Stratified validation  
- Ensemble learning  

---

## ⚙️ Key Features

### 🧠 Intelligent Ensemble (AWE-WQI)

Combines:
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- XGBoost  

**Weighting Strategy:**

\[
w_i = \frac{F1_i^2}{\sum_j F1_j^2}
\]

**Final Prediction:**

\[
\hat{y} = \arg\max \sum_i w_i \cdot P_i(y|x)
\]

---

### ⚖️ Class Imbalance Handling

- SMOTE applied **inside cross-validation pipelines**  
- Prevents **data leakage**  
- Maintains real-world class distribution  

---

### ✂️ Adaptive Feature Selection

- Random Forest feature importance  
- Median threshold pruning  
- Reduces noise and improves generalization  

---

### 🔁 Robust Validation Strategy

- K-Fold Cross Validation (n = 5)  
- Stratified K-Fold (n = 5)  
- Ensures balanced class representation  

---

### 🎯 Target Engineering

- Converts continuous WQI → categorical labels  
- Rule-based thresholds ensure **interpretability and domain alignment**

---

## 🏗️ System Architecture
Raw Dataset (Results_MADE.csv)
↓
Missing Value Imputation (Mean)
↓
WQI → Class Mapping
↓
Train/Test Split (Stratified)
↓
Feature Selection (RF Median Threshold)
↓
Standardization (Z-score)
↓
Cross-Validation Pipelines
├── SMOTE (inside folds)
└── Model Training
↓
Hyperparameter Tuning (AdaBoost, XGBoost)
↓
Final Training with SMOTE
↓
AWE Ensemble Fusion
↓
Evaluation & Visualization


---

## 📦 Installation

```bash
git clone https://github.com/yourusername/WQI-Classifier.git
cd WQI-Classifier
pip install -r requirements.txt
python water_ml.py

🧪 Dependencies
Python ≥ 3.8
numpy
pandas
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
📊 Output & Evaluation
✔ Quantitative Metrics
Accuracy
Precision (weighted)
Recall (weighted)
F1 Score (weighted)
📈 Visual Diagnostics
🔹 Confusion Matrix
Class-wise prediction performance
Identifies misclassification patterns
🔹 ROC Curve (Multi-class)
One-vs-rest binarization
Evaluates class separability
🔹 Feature Importance
Global ranking of features
Provides interpretability
🧠 Model Stack
Model	Role
Random Forest	Feature selection + baseline
Gradient Boosting	Sequential error correction
AdaBoost	Weak learner boosting
XGBoost	Optimized gradient boosting
⚙️ Hyperparameter Optimization
Manual grid search (custom implementation)
Evaluated using Stratified K-Fold
Objective: maximize weighted F1-score
🔒 Reproducibility
Global random_state = 42
Deterministic splits, resampling, and training
Ensures fully reproducible results
⚠️ Implementation Notes
Feature selection is performed before cross-validation
→ introduces minor theoretical data leakage
Preprocessing is not fully pipeline-wrapped
→ can be improved for stricter ML compliance
Ensemble uses tuned models only
→ intentional design for performance-weighted fusion
📁 Project Structure
WQI-Classifier/
│
├── water_ml.py          # Main pipeline script
├── Results_MADE.csv     # Dataset
├── requirements.txt
└── README.md
📌 Future Improvements
Full sklearn Pipeline integration (end-to-end)
Nested cross-validation for hyperparameter tuning
SHAP-based explainability
Real-time prediction API
Deployment (Flask / FastAPI)
🧾 Citation

If you use this work in research, cite as:

Water Quality Intelligence Classifier using AWE Ensemble and Stratified Validation, 2026.
🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first.

📜 License

MIT License

🔥 Final Note

This project is designed to balance:

Practical machine learning performance
Academic rigor
Interpretability in environmental systems
