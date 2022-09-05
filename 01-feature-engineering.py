# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Engineering 
# MAGIC ---
# MAGIC - Preprocessing
# MAGIC - Imputation 
# MAGIC - One-hot Encoding
# MAGIC - Feature Standardization

# COMMAND ----------

df = spark.table("hive_metastore.mchan_credit_risk_db.t1_bronze_creditcard_train")
display(df)

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

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()
