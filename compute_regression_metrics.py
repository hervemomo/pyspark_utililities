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
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.functions import vector_to_array

import logging
from functools import wraps

from pyspark.storagelevel import StorageLevel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import *
from typing import Dict, List, Optional, Tuple # for hint type
from pyspark.ml.regression import LinearRegression, LinearRegressionModel


def compute_regression_metrics(
    df: DataFrame,
    label_col: str,
    prediction_col: str,
    n_features: int
) -> Dict[str, float]:
    """
    Compute MSE, RMSE, MAE, R², and adjusted R² in just two Spark actions.
 
    Args:
        df:               Spark DataFrame containing label & prediction.
        label_col:        Name of the true‐label column.
        prediction_col:   Name of the prediction column.
        n_features:       Number of features used (for adjusted R²).
 
    Returns:
        Dict with keys: mse, rmse, mae, r2, adjusted_r2.
    """
    # 1. Validate inputs
    for c in (label_col, prediction_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame")
    if n_features < 0:
        raise ValueError(f"n_features must be non-negative (got {n_features})")
 
    # 2. Drop nulls once and cache
    clean = df.select(label_col, prediction_col).na.drop().cache()
 
    # 3. First pass: get count and label mean
    stats1 = clean.agg(
        F.count("*").alias("n"),
        F.avg(label_col).alias("label_mean")
    ).collect()[0]
    n = stats1["n"]
    if n == 0:
        raise ValueError("No rows remaining after dropping nulls")
    label_mean = stats1["label_mean"]
 
    # 4. Second pass: compute sums of squared residuals, abs errors, and total variance
    stats2 = clean.agg(
        F.sum((F.col(prediction_col) - F.col(label_col))**2).alias("ss_res"),
        F.sum(F.abs(F.col(prediction_col) - F.col(label_col))).alias("sum_abs"),
        F.sum((F.col(label_col) - F.lit(label_mean))**2).alias("ss_tot")
    ).collect()[0]
 
    ss_res = stats2["ss_res"]
    sum_abs = stats2["sum_abs"]
    ss_tot = stats2["ss_tot"]
 
    # 5. Compute metrics locally
    mse = ss_res / n
    rmse = mse**0.5
    mae = sum_abs / n
 
    # R² (perfect‐fit override if needed)
    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1 - ss_res / ss_tot
        if ss_res == 0.0:
            r2 = 1.0
 
    # Adjusted R²
    if n > n_features + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = None
 
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adjusted_r2": adj_r2
    }