import os
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import builtins
import psutil
import datetime
import time
import math # For NaN handling in summary_stats function
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel
import logging
from functools import wraps

from typing import Tuple, List # for hint type

from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, NumericType, BooleanType,
    IntegerType, FloatType, ShortType, DecimalType, TimestampType, ArrayType
)


def compute_feature_variance(df, feature_cols, variance_threshold=None):
    """
    Computes the volume, proportion and variance for binary features in a PySpark DataFrame,
    and returns a DataFrame of features with computed counts, proportions, and variance.
    Optionally, a variance_threshold can be provided to filter the features.
    
    Parameters:
    - df: PySpark DataFrame.
    - feature_cols: List of column names (assumed binary: 0 or 1).
                    Must be a non-empty list of valid column names present in df.
    - variance_threshold: (Optional) Minimum variance to retain a feature.
                          If set to None, no filtering on variance is applied.
    
    Returns:
    - DataFrame with columns: important feature, count_of_1, count_of_0,
      proportion_of_1, proportion_of_0, variance (all rounded to 4 decimals).
      
    Raises:
    - ValueError: If feature_cols is empty, contains invalid column names,
                  if the DataFrame is empty, or if variance_threshold (when provided)
                  is not a numeric type in the expected range.
    """
    
    # Validate that the feature_cols list is not empty
    if not feature_cols:
        raise ValueError("The feature_cols list is empty. Please provide a list of feature column names.")
    
    # Validate that all features exist in the DataFrame
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
    
    # Check if the DataFrame is empty
    total_count = df.count()
    if total_count == 0:
        raise ValueError("The input DataFrame is empty. Cannot compute feature variance on an empty DataFrame.")
    
    # Validate variance_threshold only if it is provided (not None)
    if variance_threshold is not None:
        if not isinstance(variance_threshold, (int, float)):
            raise ValueError("variance_threshold must be a numeric type or None.")
        if variance_threshold < 0 or variance_threshold > 0.25:
            raise ValueError("variance_threshold should be between 0 and 0.25 for binary features.") # case p=0.5
    
    # Aggregate sum of 1s for all features in feature_cols
    feature_counts_row = df.agg(*(F.sum(F.col(c)).alias(c) for c in feature_cols)).collect()[0]
    feature_counts_dict = feature_counts_row.asDict()
    
    # Use the current Spark session from the DataFrame instead of a global variable
    spark_session = df.sparkSession
    
    # Create a new DataFrame from the feature counts dictionary
    data = [(k, v) for k, v in feature_counts_dict.items()]
    narrow_df = spark_session.createDataFrame(data, schema=["feature", "count_of_1"])
    
    # Add columns for count_of_0, proportions, and variance
    narrow_df = narrow_df.withColumn("count_of_0", F.lit(total_count) - F.col("count_of_1")) \
                         .withColumn("proportion_of_1", F.col("count_of_1") / F.lit(total_count)) \
                         .withColumn("proportion_of_0", F.col("count_of_0") / F.lit(total_count)) \
                         .withColumn("variance", F.col("proportion_of_1") * (1 - F.col("proportion_of_1"))) \
                         .withColumn("proportion_of_1", F.round("proportion_of_1", 4)) \
                         .withColumn("proportion_of_0", F.round("proportion_of_0", 4)) \
                         .withColumn("variance", F.round("variance", 4))
    
    # If a variance_threshold is provided, filter the DataFrame accordingly
    if variance_threshold is not None:
        narrow_df = narrow_df.filter(F.col("variance") >= variance_threshold)
    
    print('Variance calculation completed')
    
    return narrow_df