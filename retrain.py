import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure the api directory is in the path
ROOT = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(ROOT, "api")
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from churnexplainer import ExplainedModel, CategoricalEncoder

print("--- Starting Local Model Re-training ---")

# 1. Load Data
data_path = os.path.join(ROOT, "raw", "WA_Fn-UseC_-Telco-Customer-Churn-.csv")
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# 2. Basic Cleaning (Match original logic)
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# 3. Features and Labels
labels = df["Churn"].map({"Yes": 1, "No": 0})
data = df.drop("Churn", axis=1)

# Categorical columns
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    data[col] = data[col].astype("category")

train_df, test_df, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# 4. Train Model
print("Training Logistic Regression model...")
ce = CategoricalEncoder().fit(train_df)
X_train = ce.transform(train_df)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 5. Build Explanation
print("Initializing LIME explainer (this might take a moment)...")
em = ExplainedModel(
    data=train_df,
    labels=train_labels,
    categoricalencoder=ce,
    pipeline=model
)

# Initialize the explainer internally
from lime.lime_tabular import LimeTabularExplainer
em.explainer = LimeTabularExplainer(
    ce.transform(train_df),
    feature_names=list(train_df.columns),
    class_names=["No Churn", "Churn"],
    categorical_features=list(ce.cat_columns_ix_.values()),
    categorical_names=ce.classes_,
    discretize_continuous=True,
)

# 6. Save
print("Saving fresh model to models/telco_linear/telco_linear.pkl...")
em.save("telco_linear")

print("\nSUCCESS! New compatible model generated.")
print("Now push this file to GitHub to fix Vercel.")
