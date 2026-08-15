# Forest Fire Risk Prediction using Ensemble Machine Learning

## 1. Project Overview

This project develops an end-to-end machine learning solution for the
**Earth / Environment** mission of the ML for Social Good Challenge.

The objective is to predict whether environmental conditions indicate
**Forest Fire** or **No Fire** using ensemble machine learning.

The final selected model is **XGBoost**, supported by SHAP-based
explainability.

---

## 2. Problem Statement

Forest fires can cause significant environmental and economic damage.
Early identification of high-risk environmental conditions can support
monitoring and preventive action.

This project uses historical environmental and fire-weather observations to
classify conditions as:

- **Fire**
- **No Fire**

The intended users are environmental monitoring teams and other stakeholders
who can use the prediction as an early-warning decision-support signal.

The model is not intended to replace human or emergency decision-making.

---

## 3. Dataset

The project uses the **Algerian Forest Fires dataset** containing
environmental and fire-weather observations.

The dataset includes variables such as:

- Temperature
- Relative Humidity
- Wind Speed
- Rain
- FFMC
- DMC
- DC
- ISI
- BUI
- FWI

Additional features were derived during preprocessing and feature engineering.

The target variable is:

- `Fire` = Fire condition
- `No Fire` = No Fire condition

Dataset source:

**Algerian Forest Fires Dataset, UCI Machine Learning Repository.**

---

## 4. Methodology

The notebook follows the complete machine learning workflow:

1. Data loading and cleaning
2. Missing-value and duplicate checks
3. Invalid-value and outlier analysis
4. Exploratory Data Analysis
5. Feature engineering
6. Leakage-safe preprocessing
7. Group-aware train/validation/test splitting
8. Model training and tuning
9. Ensemble comparison
10. Final test evaluation
11. SHAP explainability
12. Synthetic live prediction

---

## 5. Models

Four models were evaluated:

1. **Decision Tree** — baseline model
2. **Random Forest** — bagging ensemble
3. **XGBoost** — boosting ensemble
4. **Stacking Ensemble** — heterogeneous ensemble combining
   Decision Tree, Random Forest and XGBoost

Cross-validation was used during model tuning, and leakage-safe
out-of-fold predictions were used for the stacking meta-learner.

---

## 6. Final Test Results

All models were evaluated on the same untouched test set.

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 0.8663 | 0.8305 | 0.8829 | 0.8559 |
| Random Forest | 0.9946 | 0.9145 | 0.9640 | 0.9386 |
| XGBoost | **0.9963** | **0.9469** | 0.9640 | **0.9554** |
| Stacking | 0.9958 | 0.9237 | **0.9820** | 0.9520 |

**XGBoost was selected as the final model** because it achieved the highest
ROC-AUC and F1-score while maintaining high Fire-class recall.

---

## 7. Explainability

SHAP was used to explain the final XGBoost model at two levels:

- **Global explanation:** identifies the features with the greatest overall
  influence on predictions.
- **Individual explanation:** explains the factors contributing to a
  specific prediction.

FWI and other fire-weather variables were among the most influential
features.

---

## 8. Live Demonstration

A realistic synthetic environmental record was passed through the trained
XGBoost model to demonstrate a new prediction.

The demonstration produced a:

**Fire Risk probability: 99.1%**

A local SHAP explanation was also generated to show the factors contributing
to this prediction.

The synthetic record was used only for demonstration and was not added to
the training or test data.

---

## 9. Ethics and Limitations

The model is intended as a **decision-support and early-warning tool**, not
an autonomous emergency system.

Important limitations include:

- False negatives may result in missed fire-risk conditions.
- False positives may result in unnecessary monitoring.
- Predictions may be less reliable for conditions outside the training data.
- Geographic and temporal differences may affect model performance.
- The model should be externally validated before operational deployment.
- Human and domain-expert oversight is required.

The project uses environmental observations and does not require personally
identifiable information.

---

## 10. How to Run

### Requirements

Python 3.x and the libraries used in the notebook are required.

The main libraries include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap

### Steps

1. Download or clone the project.
2. Install the required Python libraries.
3. Place the dataset in the location specified in the notebook.
4. Open the Jupyter Notebook.
5. Run the cells sequentially from beginning to end.

The notebook contains the complete preprocessing, training, evaluation,
ensemble modelling, explainability, and live prediction workflow.

---

## 11. Files

- `2547262_ForestFire.ipynb` — complete ML implementation
- `README.md` — project and execution information
- `requirements.txt` — Python dependencies
- `models/` — trained model/pipeline, if included
- `figures/` — selected results and explainability figures, if included

---

## 12. Dataset Citation

Dataset:

**Algerian Forest Fires Dataset**  
UCI Machine Learning Repository.

The dataset is publicly available and was used for academic purposes in this
project.