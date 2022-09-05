# Databricks notebook source
# MAGIC %md
# MAGIC ## Environment Recreation

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os

model_name = "mchan_creditcard_random_forest_model"
model_uri = f"models:/{model_name}/1"
local_path = ModelsArtifactRepository(model_uri).download_artifacts("") # download model from remote registry

requirements_path = os.path.join(local_path, "requirements.txt")
if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

# COMMAND ----------

# MAGIC %pip install -r $requirements_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define input and output

# COMMAND ----------

model_name = "mchan_creditcard_random_forest_model"
input_table = "hive_metastore.mchan_credit_risk_db.t4_customers_to_be_rated_final"
output_table = "/FileStore/batch-inference/mchan_creditcard_random_forest_model_output"

# COMMAND ----------

df = spark.table(input_table)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the ML Model

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct

model_uri = f"models:/{model_name}/1"

# create spark user-defined function for model prediction
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double", env_manager = "conda")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Batch Inference on New Customers

# COMMAND ----------

output_df = df.withColumn("prediction", predict(struct(*df.columns)))
display(output_df)

# COMMAND ----------

output_df.write.saveAsTable("hive_metastore.mchan_credit_risk_db.t5_customers_rated_predictions")
