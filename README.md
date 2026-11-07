# Predictive Modeling for Clinical Diabetes Risk
## An End-to-End Machine Learning Pipeline, Optimization & SHAP Interpretability Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Interpretability-SHAP-yellow.svg)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning pipeline built to predict the onset of diabetes mellitus using physiological and clinical indicators from the benchmark **Pima Indians Diabetes Database** (NIDDK). This project implements rigorous data preprocessing—including zero-masking, MICE imputation, and domain-driven feature engineering—coupled with an optimized XGBoost classifier and game-theoretic SHAP interpretability.

---

## 🚀 Key Features & Pipeline Architecture

The repository contains a robust, production-grade ML pipeline structured into clear logical layers:

1. **Zero-Masking & Missingness Preservation:** 
   * Automatically identifies biologically impossible zero-values across clinical attributes (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`).
   * Constructs explicit binary missingness indicator features (`*_missing`) to retain pattern information.
2. **Advanced Multivariate Imputation (MICE):** 
   * Implements `IterativeImputer` utilizing Bayesian Ridge regression to impute missing physiological parameters conditionally.
3. **Domain-Driven Feature Engineering:** 
   * Synthesizes clinically relevant interaction variables, including the `Glucose_BMI_Interaction` index and epidemiological `Age_Risk_Group` brackets.
4. **Outlier Treatment & Normalization:** 
   * Applies percentile-based Winsorization (1st and 99th percentiles) to stabilize gradient updates, followed by Gaussian regularization via `QuantileTransformer`.
5. **Rigorous Hyperparameter Tuning:** 
   * Evaluated via 5-fold `StratifiedKFold` cross-validation combined with `GridSearchCV` optimized for `roc_auc`.
6. **Model Interpretability (SHAP):** 
   * Integrates `TreeExplainer` to compute exact Shapley values for global and local decision transparency.

---

## 📊 Performance & Evaluation Results

Tested on a held-out evaluation split ($N=154$), the optimized model demonstrates robust discriminative performance:

| Metric | Baseline Logistic Regression | Standard Random Forest | **Optimized XGBoost Pipeline** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 71.43% | 72.08% | **73.38%**[cite: 5] |
| **Precision (Class 1)** | 0.6042 | 0.6122 | **0.6383**[cite: 5] |
| **Recall / Sensitivity** | 0.5370 | 0.5556 | **0.5556**[cite: 5] |
| **F1-Score (Class 1)** | 0.5686 | 0.5825 | **0.5941**[cite: 5] |
| **ROC-AUC Score** | 0.7712 | 0.7945 | **0.8202**[cite: 5] |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Core Libraries:** 
  * `pandas`, `numpy` (Data manipulation & numerical operations)
  * `scikit-learn` (Preprocessing, MICE imputation, cross-validation, metrics)
  * `xgboost` (Gradient boosting classification engine)
  * `shap` (Model explanation and interpretability plots)
  * `matplotlib`, `seaborn` (Data visualization & diagnostic plotting)

---

## 📂 Project Structure

```text
├── data/                      # Dataset files (Pima Indians Diabetes Database)
├── notebooks/                 # Jupyter Notebooks containing end-to-end implementation
│   └── diabetes_prediction.ipynb
├── reports/                   # Comprehensive written technical reports & documentation
│   └── Task_3(Doc)_3.docx
├── README.md                  # Project documentation
└── requirements.txt           # Python package dependencies
