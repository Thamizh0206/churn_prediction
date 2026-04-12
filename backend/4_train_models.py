# =============================================================================
# Telco Customer Churn Prediction
# Script : 4_train_models.py
# Purpose: Train a logistic regression model with LIME explainability and
#          serialize the complete ExplainedModel artifact to disk.
# License: Apache 2.0 (Original dataset: IBM Sample Data)
# =============================================================================
# To simply train the model once, run this file in a workbench session.
#
# There are 2 other ways of running the model training process
#
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model
# training process as a job, create a new job by going to the Project window and clicking _Jobs >
# New Job_ and entering the following settings:
# * **Name** : Train Mdoel
# * **Script** : 4_train_models.py
# * **Arguments** : _Leave blank_
# * **Kernel** : Python 3
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 2 GiB
# The rest can be left as is. Once the job has been created, click **Run** to start a manual
# run for that job.

# ***Experiments***
#
# Training a model for use in production requires testing many combinations of model parameters
# and picking the best one based on one or more metrics.
# In order to do this in a *principled*, *reproducible* way, an Experiment executes model training code with **versioning** of the **project code**, **input parameters**, and **output artifacts**.
# This is a very useful feature for testing a large number of hyperparameters in parallel on elastic cloud resources.

# **[Experiments](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-running-an-experiment.html)**.
# run immediately and are used for testing different parameters in a model training process.
# In this instance it would be use for hyperparameter optimisation. To run an experiment, from the
# Project window click Experiments > Run Experiment with the following settings.
# * **Script** : 4_train_models.py
# * **Arguments** : 5 lbfgs 100 _(these the cv, solver and max_iter parameters to be passed to
# LogisticRegressionCV() function)
# * **Kernel** : Python 3
# * **Engine Profile** : 1 vCPU / 2 GiB

# Click **Start Run** and the expriment will be sheduled to build and run. Once the Run is
# completed you can view the outputs that are tracked with the experiment using the
# `cml.metrics_v1.track_metrics` function. It's worth reading through the code to get a sense of what
# all is going on.

# More Details on Running Experiments
# Requirements
# Experiments have a few requirements:
# - model training code in a `.py` script, not a notebook
# - `requirements.txt` file listing package dependencies
# - a `cdsw-build.sh` script containing code to install all dependencies
#
# These three components are provided as `4_train_models.py`, `requirements.txt`,
# and a shell build script, respectively.
# You can see that the build script simply installs packages from `requirements.txt`.
# The code in `4_train_models.py` is largely identical to the code in the last notebook.
# with a few differences.
#
# The first difference from the last notebook is at the "Experiments options" section.
# When you set up a new Experiment, you can enter
# [**command line arguments**](https://docs.python.org/3/library/sys.html#sys.argv)
# in standard Python fashion.
# This will be where you enter the combination of model hyperparameters that you wish to test.
#
# The other difference is at the end of the script.
# Here, the `cdsw` package (available by default) provides
# [two methods](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-tracking-metrics.html)
# to let the user evaluate results.
#
# **`cml.metrics_v1.track_metric`** stores a single value which can be viewed in the Experiments UI.
# Here we store two metrics and the filepath to the saved model.
#


import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from lime.lime_tabular import LimeTabularExplainer

try:
  os.chdir("code")
except:
  pass
from churnexplainer import ExplainedModel, CategoricalEncoder

labelcol = "Churn"

# Read directly from raw CSV local file
csv_path = os.path.join("../raw", "WA_Fn-UseC_-Telco-Customer-Churn-.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join("raw", "WA_Fn-UseC_-Telco-Customer-Churn-.csv")
df = pd.read_csv(csv_path)


# Clean and prep the dataframe
df = (df
      .replace(r"^\s$", np.nan, regex=True).dropna().reset_index()
      # drop unnecessary and personally identifying information
      .drop(columns=['index', 'customerID'])
     )

df['SeniorCitizen'] = df['SeniorCitizen'].astype(str).replace({
    "1": "Yes",
    "0": "No",
})
  
df['TotalCharges'] = df['TotalCharges'].astype('float')
df.index.name='id'


# separate target variable column from feature columns
datadf, labels = df.drop(labelcol, axis=1), df[labelcol]

# recast all columns that are "object" dtypes to Categorical
for colname, dtype in zip(datadf.columns, datadf.dtypes):
  if dtype == "object":
    datadf[colname] = pd.Categorical(datadf[colname])

  
# Prepare data for Sklearn model and create train/test split
ce = CategoricalEncoder()
X = ce.fit_transform(datadf)
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ct = ColumnTransformer(
    [("ohe", OneHotEncoder(), list(ce.cat_columns_ix_.values()))],
    remainder="passthrough",
)

# Experiments options
# If you are running this as an experiment, pass the cv, solver and max_iter values
# as arguments in that order. e.g. `5 lbfgs 100`.
if len(sys.argv) == 4:
    try:
        cv = int(sys.argv[1])
        solver = str(sys.argv[2])
        max_iter = int(sys.argv[3])
    except:
        sys.exit("Invalid Arguments passed to Experiment")
else:
    cv = 5
    solver = "lbfgs"  # one of newton-cg, lbfgs, liblinear, sag, saga
    max_iter = 100

# Instantiate the model
clf = LogisticRegressionCV(cv=cv, solver=solver, max_iter=max_iter)
pipe = Pipeline([("ct", ct), ("scaler", StandardScaler()), ("clf", clf)])

# Train the model
pipe.fit(X_train, y_train)

# Capture train and test set scores
train_score = pipe.score(X_train, y_train)
test_score = pipe.score(X_test, y_test)
print("train", train_score)
print("test", test_score)
print(classification_report(y_test, pipe.predict(X_test)))
datadf[labels.name + " probability"] = pipe.predict_proba(X)[:, 1]


# Create LIME Explainer
feature_names = list(ce.columns_)
categorical_features = list(ce.cat_columns_ix_.values())
categorical_names = {i: ce.classes_[c] for c, i in ce.cat_columns_ix_.items()}
class_names = ["No " + labels.name, labels.name]
explainer = LimeTabularExplainer(
    ce.transform(datadf),
    feature_names=feature_names,
    class_names=class_names,
    categorical_features=categorical_features,
    categorical_names=categorical_names,
)


# Create and save the combined Logistic Regression and LIME Explained Model.
explainedmodel = ExplainedModel(
    data=datadf,
    labels=labels,
    categoricalencoder=ce,
    pipeline=pipe,
    explainer=explainer,
)
explainedmodel.save(model_name='telco_linear')


# Print metrics for local training run
print("Final Train Score:", round(train_score, 2))
print("Final Test Score:", round(test_score, 2))
# Wrap up

# We've now covered all the steps to **running Experiments**.
#
# Notice also that any script that will run as an Experiment can also be run as a Job or in a Session.
# Our provided script can be run with the same settings as for Experiments.
# A common use case is to **automate periodic model updates**.
# Jobs can be scheduled to run the same model training script once a week using the latest data.
# Another Job dependent on the first one can update the model parameters being used in production
# if model metrics are favorable.
