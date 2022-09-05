# Databricks notebook source
featurized_df = spark.table("hive_metastore.mchan_credit_risk_db.t2_feature_store_table")
display(featurized_df)

# COMMAND ----------

from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS mchan_creditrisk_feature_store; 
# MAGIC USE mchan_creditrisk_feature_store; 

# COMMAND ----------

featureStore = feature_store.FeatureStoreClient()

featureStore.create_table(
    name = "mchan_creditrisk_feature_store.creditcard_default_features",
    primary_keys = ["row_id"],
    df = featurized_df, 
    description = "Feature table to predict creditcard default"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### -- END OF STEP 02 -- 
