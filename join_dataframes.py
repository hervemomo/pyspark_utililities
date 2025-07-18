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

def join_dataframes(df1: DataFrame, df2: DataFrame, keys: list, how: str = "inner") -> DataFrame:
    """
    Perform an optimized join between two PySpark DataFrames.

    Parameters:
        df1 (DataFrame): The first DataFrame.
        df2 (DataFrame): The second DataFrame.
        keys (list): List of column names to join on.
        how (str): Type of join ('inner', 'left', 'right', 'full'). Default is 'inner'.

    Returns:
        DataFrame: The joined DataFrame with duplicate key columns from df2 dropped.
    """
    import time
    from pyspark.sql import functions as F

    start_time = time.time()

    # Edge Case 1: Validate Inputs
    if not isinstance(df1, DataFrame) or not isinstance(df2, DataFrame):
        raise TypeError("Both df1 and df2 must be PySpark DataFrames.")
    
    if not isinstance(keys, list) or len(keys) == 0:
        raise ValueError("Keys must be a non-empty list of column names.")

    valid_joins = {"inner", "left", "right", "full", "outer", "left_outer", "right_outer", "full_outer"}
    if how.lower() not in valid_joins:
        raise ValueError(f"Invalid join type '{how}'. Supported joins: {valid_joins}")

    # Edge Case 2: Standardize Column Names (Strip Spaces & Lowercase)
    df1 = df1.toDF(*[c.strip().lower() for c in df1.columns])
    df2 = df2.toDF(*[c.strip().lower() for c in df2.columns])
    keys = [key.strip().lower() for key in keys]  # Normalize keys too

    # Edge Case 3: Ensure Join Keys Exist in Both DataFrames
    missing_keys_df1 = [k for k in keys if k not in df1.columns]
    missing_keys_df2 = [k for k in keys if k not in df2.columns]

    if missing_keys_df1 or missing_keys_df2:
        raise ValueError(f"Missing join keys:\n"
                         f"  - In df1: {missing_keys_df1}\n"
                         f"  - In df2: {missing_keys_df2}")


    # Identify common columns between df1 and df2, excluding join keys
    common_cols = set(df1.columns) & set(df2.columns) - set(keys)
    
    # Drop common columns from df2 except the join keys
    df2 = df2.drop(*common_cols)
    print(f'{len(common_cols)} Common columns deleted from the right dataframe:')
    print(common_cols)
    print()


    # Rename duplicate key columns in df2 to avoid collision.
    # Each key in df2 is renamed to "<key>_dup".
    for k in keys:
        df2 = df2.withColumnRenamed(k, f"{k}_dup")

    # Build join conditions: match df1.key with df2.key_dup for each key.
    join_conditions = [df1[k] == df2[f"{k}_dup"] for k in keys]

    # Perform the join using the conditions
    df = df1.join(df2, on=join_conditions, how=how.lower())

    # After the join, drop the duplicate key columns from df2.
    duplicate_columns = [f"{k}_dup" for k in keys]
    df = df.drop(*duplicate_columns)

    # Edge Case 5: Handle Empty Join Results
    if df.count() == 0:
        print("Warning: The join resulted in an empty DataFrame.")

    return df