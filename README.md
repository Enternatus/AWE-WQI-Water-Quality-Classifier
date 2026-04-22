💧 Water Quality Intelligence (WQI) Classifier

A robust, leakage-aware machine learning pipeline for multi-class water quality classification, combining feature selection, imbalance handling, and a custom Accuracy-Weighted Ensemble (AWE) for high-fidelity environmental prediction.

🚀 Overview

This project implements an end-to-end supervised learning system that classifies water samples into five standardized quality categories:

Class	WQI Range
Excellent	0 – 25
Good	26 – 50
Moderate	51 – 75
Poor	76 – 100
Very Poor	> 100

The system integrates:

Statistical preprocessing
Embedded feature selection
Leakage-free resampling
Stratified validation
Ensemble learning
⚙️ Key Features
🧠 Intelligent Ensemble (AWE-WQI)
Combines Random Forest, Gradient Boosting, AdaBoost, XGBoost

	​

(y∣x)
⚖️ Class Imbalance Handling
Uses SMOTE inside cross-validation pipelines
Prevents data leakage
Preserves real-world distribution in validation folds
✂️ Adaptive Feature Selection
Random Forest importance ranking
Median threshold pruning
Reduces noise + improves generalization
🔁 Robust Validation Strategy
Standard K-Fold (n=5)
Stratified K-Fold (n=5) for class balance
Ensures minority class representation
🎯 Target Engineering
Converts continuous WQI into discrete classes
Rule-based thresholds ensure interpretability + domain alignment
🏗️ System Architecture
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
📦 Installation
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
Detects misclassification patterns
🔹 ROC Curve (Multi-class)
One-vs-rest binarization
Evaluates separability
🔹 Feature Importance
Global ranking of input variables
Interpretable model insights
🧠 Model Stack
Model	Role
Random Forest	Feature selection + baseline
Gradient Boosting	Sequential error correction
AdaBoost	Weak learner boosting
XGBoost	High-performance optimized boosting
⚙️ Hyperparameter Optimization
Manual grid search (custom implementation)
Optimized using Stratified K-Fold
Objective: maximize weighted F1-score
🔒 Reproducibility
Global random_state = 42
Deterministic splits, resampling, and training
Ensures consistent academic results
⚠️ Implementation Notes
Feature selection is performed before cross-validation
→ minor theoretical leakage (acceptable but improvable)
Preprocessing is not fully pipeline-wrapped
→ could be refactored into a unified pipeline for stricter ML compliance
Ensemble uses tuned models only
→ intentional for performance weighting
📁 Project Structure
WQI-Classifier/
│
├── water_ml.py          # Main pipeline script
├── Results_MADE.csv     # Dataset
├── requirements.txt
└── README.md
📌 Future Improvements
Full sklearn Pipeline integration (end-to-end)
Nested cross-validation for tuning
SHAP-based explainability
Real-time prediction API
Deployment (Flask / FastAPI)
🧾 Citation (for Academic Use)

If you use this work in research, cite as:

Water Quality Intelligence Classifier using AWE Ensemble and Stratified Validation, 2026.
🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

📜 License

MIT License

🔥 Final Note

This project is designed to balance:

Practical ML performance
Academic rigor
Interpretability in environmental systems
