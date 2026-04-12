# Code Directory — Step-by-Step Run Guide

This directory contains the complete ML pipeline. Follow the steps below **in order**.

---

## Prerequisites

Make sure you have completed the setup from the root `README.md`:
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Step 1 — Explore the Data (Optional)

Open the EDA notebook to understand the dataset before training:

```
2_data_exploration.ipynb
```

Launch with:
```bash
jupyter notebook code/2_data_exploration.ipynb
```

**What it covers:**
- Distribution of churn vs. non-churn customers (~26.5% churn rate)
- Feature distributions: `tenure`, `MonthlyCharges`, `TotalCharges`
- Churn rate breakdown by `Contract`, `InternetService`, `PaymentMethod`
- Correlation heatmap

---

## Step 2 — Understand the Model (Optional)

Open the model building notebook to walk through the full pipeline step by step:

```
3_model_building.ipynb
```

**What it covers:**
- Data cleaning and encoding
- Building the sklearn `Pipeline`
- Training `LogisticRegressionCV`
- Generating LIME explanations
- Visualizing feature importances

---

## Step 3 — Train the Model ⭐

```bash
python code/4_train_models.py
```

**What it does:**
1. Reads `raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv` directly
2. Cleans and preprocesses the data
3. Encodes categorical features using `CategoricalEncoder`
4. Trains a `LogisticRegressionCV` pipeline
5. Builds a `LimeTabularExplainer` on all training data
6. Saves the complete `ExplainedModel` to `models/telco_linear/telco_linear.pkl`

**Expected output:**
```
train 0.8075
test  0.7918
              precision    recall  f1-score   support
          No       0.84      0.89      0.86      1300
         Yes       0.62      0.52      0.57       458
    accuracy                           0.79      1758

Final Train Score: 0.81
Final Test Score: 0.79
```

> You can also pass CLI arguments to override hyperparameters:
> ```bash
> python code/4_train_models.py 5 lbfgs 100
> # Arguments: <cv_folds> <solver> <max_iter>
> ```

---

## Step 4 — Launch the Web Application ⭐

```bash
python code/6_application.py
```

Open your browser at: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

### Available pages:

| Page | URL | Description |
|---|---|---|
| Table View | `/flask/table_view.html` | 10 random customers, color-coded by churn risk and LIME impact |
| Single View | `/flask/single_view.html?...` | One customer's prediction + all feature-level LIME explanations |

### API Endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/model` | POST | Run inference on a customer JSON dict |
| `/sample_table` | GET | 10 random explained customers |
| `/categories` | GET | All categorical feature class options |
| `/stats` | GET | Numerical feature statistics |

---

## File Reference

| File | Purpose |
|---|---|
| `4_train_models.py` | **Main training script** — run this to build the model |
| `5_model_serve_explainer.py` | Contains the `explain()` function logic used by the Flask app |
| `6_application.py` | **Flask app** — web server, routes, and local `/model` inference endpoint |
| `churnexplainer.py` | `ExplainedModel` and `CategoricalEncoder` classes — the core ML helper module |
| `2_data_exploration.ipynb` | EDA notebook |
| `3_model_building.ipynb` | Model development walkthrough notebook |
| `7a_ml_ops_simulation.py` | MLOps drift simulation (reference only — requires cloud environment) |
| `7b_ml_ops_visual.py` | MLOps drift visualization plots (reference only) |

---

## Understanding `churnexplainer.py`

This is the most important module in the project.

### `CategoricalEncoder`

A custom `sklearn` transformer that:
- Fits a `LabelEncoder` on each categorical column
- Transforms a mixed-type DataFrame into a float numpy array (integer codes)
- Required because LIME needs integer-coded categoricals, not one-hot encoded

### `ExplainedModel`

A container class that bundles together:
- `data` — the full training DataFrame (with predicted probabilities)
- `labels` — the `Churn` target Series
- `categoricalencoder` — fitted `CategoricalEncoder`
- `pipeline` — fitted sklearn `Pipeline` (OHE → Scaler → LogisticRegressionCV)
- `explainer` — fitted `LimeTabularExplainer`

**Key methods:**
- `save(model_name)` → serializes to `../models/<name>/<name>.pkl` using `dill`
- `load(model_name)` → loads and returns a ready-to-use `ExplainedModel`
- `explain_dct(dict)` → takes a customer dict, returns `(probability, explanation_dict)`
- `cast_dct(dict)` → normalizes a customer dict's value types to match training dtypes

---

## Model Architecture Summary

```
Raw DataFrame (19 features)
        │
        ▼
CategoricalEncoder.fit_transform()
→ int-coded numpy array  [used by LIME]
        │
        ▼
sklearn Pipeline:
    ColumnTransformer
        └── OneHotEncoder on categorical indices
        └── passthrough for numeric columns
    StandardScaler
    LogisticRegressionCV(cv=5, solver='lbfgs', max_iter=100)
        │
        ▼
    predict_proba() → Churn probability [0.0 – 1.0]
        │
        ▼
LimeTabularExplainer.explain_instance()
→ {feature: impact_weight, ...}
```
