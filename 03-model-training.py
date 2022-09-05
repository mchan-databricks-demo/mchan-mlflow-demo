# Databricks notebook source
# MAGIC %md
# MAGIC # Random Forest training
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/2191467597452823/s?orderByKey=metrics.%60val_roc_auc_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/2191467597452817) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _10.4.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "default"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load input data into a pandas DataFrame.
import pandas as pd
df_loaded = pd.read_parquet("file:///dbfs/mchan-credit-risk-demo/22-09-05-13:34/7eded5b0")

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["age_f", "has_defaulted_f", "education_f", "gender_f", "num_credit_cards_f", "annual_income_f"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["age_f", "annual_income_f"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers, sparse_threshold=0))
])

transformers.append(("numerical", numerical_pipeline, ["age_f", "annual_income_f"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, sparse_threshold=0, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers.append(("onehot", one_hot_pipeline, ["education_f", "gender_f", "has_defaulted_f", "num_credit_cards_f"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=176090223, stratify=split_y)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=176090223, stratify=split_y_rem)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/2191467597452823/s?orderByKey=metrics.%60val_roc_auc_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
help(RandomForestClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display="diagram")

skrf_classifier = RandomForestClassifier(
  bootstrap=False,
  criterion="entropy",
  max_depth=11,
  max_features=0.6459261993205776,
  min_samples_leaf=0.30636022606018404,
  min_samples_split=0.392730967640084,
  n_estimators=6,
  random_state=176090223,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", skrf_classifier),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(experiment_id="2191467597452823", run_name="random_forest") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    skrf_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    skrf_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    skrf_val_metrics = {k.replace("val_", ""): v for k, v in skrf_val_metrics.items()}
    skrf_test_metrics = {k.replace("test_", ""): v for k, v in skrf_test_metrics.items()}
    display(pd.DataFrame([skrf_val_metrics, skrf_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, len(X_train.index)))

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

model_name = "mchan_creditcard_random_forest_model"
model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
mlflow.register_model(model_uri, model_name)
