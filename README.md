# Telco Customer Churn Prediction

> A complete end-to-end machine learning system for predicting customer churn in the telecommunications industry, with real-time LIME-based model explanations and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?logo=flask)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Locally%20Runnable-brightgreen)

---

## Overview

This project builds a **Logistic Regression** classification model to predict the probability that a telecommunications customer will churn. It goes beyond simple prediction — every decision made by the model is explained using **LIME (Local Interpretable Model-agnostic Explanations)**, giving business users a clear picture of *which features drove each individual prediction*.

The system is served through a **Flask web application** that allows any user to:
- View a dashboard of randomly sampled customers with color-coded churn risk
- Drill into any individual customer to see their prediction and feature-level explanations
- Run interactive **what-if analysis** by changing feature values and seeing the prediction update in real time

---

## Live Demo Screenshots

| Table View — Global Dashboard | Single Customer — LIME Explanation |
|---|---|
| 10 random customers sorted by churn risk. Red = high risk, Blue = low risk | Per-feature LIME impact scores with interactive what-if controls |

---

## Key Features

- ✅ **Logistic Regression with Cross-Validated Regularization** (`LogisticRegressionCV`)
- ✅ **LIME Explainability** — local feature attribution for every prediction
- ✅ **Interactive What-If Analysis** — change any customer feature, see updated probability instantly
- ✅ **Full sklearn Pipeline** — OHE → StandardScaler → LogisticRegressionCV chained cleanly
- ✅ **Custom `ExplainedModel` class** — bundles model, encoder, explainer, and data into one artifact
- ✅ **Flask REST API** — `/model` endpoint accepts customer JSON and returns prediction + explanation
- ✅ **D3.js Frontend** — data-driven, color-coded UI with no heavy JavaScript framework
- ✅ **100% locally runnable** — no cloud infrastructure required

---

## Project Structure

```
CML_AMP_Churn_Prediction/
│
├── code/                          # Core ML pipeline
│   ├── 4_train_models.py          # ★ Model training — run this first
│   ├── 5_model_serve_explainer.py # Model serving logic (used by Flask)
│   ├── 6_application.py           # ★ Flask web application — run this to launch UI
│   ├── churnexplainer.py          # ExplainedModel + CategoricalEncoder classes
│   ├── 2_data_exploration.ipynb   # EDA Jupyter notebook
│   ├── 3_model_building.ipynb     # Model development notebook
│   └── README.md                  # Step-by-step run guide
│
├── flask/                         # Frontend assets
│   ├── table_view.html            # Global customer dashboard
│   ├── single_view.html           # Individual customer prediction + LIME view
│   ├── churn_vis.css              # Stylesheet
│   └── ajax-loader.gif            # Loading spinner
│
├── raw/                           # Dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn-.csv   # IBM Telco Churn dataset (7,043 rows)
│
├── models/                        # Created at runtime after training
│   └── telco_linear/telco_linear.pkl
│
├── src/                           # API utility module
├── model_metrics.db               # Pre-seeded SQLite model drift metrics
├── requirements.txt               # Python dependencies
└── setup.py                       # Package definition
```

---

## Dataset

**Source:** IBM Watson Analytics — Telco Customer Churn  
**Size:** 7,043 customers × 21 features  
**Target:** `Churn` (Yes / No)

Key features: `tenure`, `Contract`, `InternetService`, `MonthlyCharges`, `TotalCharges`, and 16 other service-related fields.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| ML Model | scikit-learn 1.3.2 (`LogisticRegressionCV`) |
| Explainability | LIME 0.2.0.1 |
| Data | pandas 2.1.4, NumPy 1.26.4 |
| Serialization | dill 0.3.8 |
| Web Backend | Flask 3.0.3 |
| Web Frontend | D3.js v5, Lodash 4.17.11 |
| Visualization | seaborn 0.13.2 |
| Database | SQLite (model_metrics.db) |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python code/4_train_models.py
```

Expected output:
```
train 0.8075
test  0.7918
Final Train Score: 0.81
Final Test Score: 0.79
```

This saves the model to `models/telco_linear/telco_linear.pkl`.

### 5. Launch the web application

```bash
python code/6_application.py
```

Open your browser at: **http://127.0.0.1:5000/**

---

## Model Performance

| Metric | Value |
|---|---|
| Train Accuracy | 81% |
| Test Accuracy | 79% |
| Precision (Churn) | 0.62 |
| Recall (Churn) | 0.52 |
| F1-Score (Weighted) | 0.79 |

---

## API Reference

### `POST /model`

Run inference on a customer profile.

**Request body:**
```json
{
  "request": {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "tenure": 12,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }
}
```

**Response:**
```json
{
  "response": {
    "prediction": {
      "data": { ... },
      "probability": 0.73,
      "explanation": {
        "InternetService": 0.20,
        "Contract": 0.14,
        "tenure": -0.11
      }
    }
  }
}
```

### `GET /sample_table`
Returns 10 randomly explained customers as a JSON array.

### `GET /categories`
Returns all categorical feature options.

### `GET /stats`
Returns numerical feature statistics (mean, median, min, max, std).

---

## How LIME Works Here

For each customer prediction, LIME:
1. **Perturbs** the input — generates synthetic variants of the customer's data
2. **Weights** variants by distance from the original
3. **Fits a local linear model** on the weighted variants
4. Returns the **linear coefficients** as feature importance scores

These scores are displayed as color bars:
- 🔴 **Red / Orange** → Feature increases churn probability
- 🔵 **Blue** → Feature decreases churn probability

---

## License

Apache 2.0 — see [LICENSE.txt](LICENSE.txt) for details.

Original dataset: IBM Sample Data (publicly available).