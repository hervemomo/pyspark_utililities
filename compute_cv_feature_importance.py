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



def compute_cv_feature_importance(cv_df, coef_threshold=1e-6):
    """
    Computes feature importance summary from submodel coefficients.

    Args:
        cv_df: Spark DataFrame from extract_cv_submodel_coefficients_with_intercept()
        coef_threshold: Coefficients with absolute value below this are treated as zero

    Returns:
        Spark DataFrame with feature importance metrics:
        feature, mean_coef, std_coef, coef_var,
        mean_abs_coef, min_coef, max_coef,
        freq_nonzero, freq_zero,
        total_obs, nonzero_ratio, zero_ratio
    """
    # Preprocess
    prepped_df = (
        cv_df
        .withColumn("abs_coef", F.abs("coefficient"))
        .withColumn("nonzero", F.when(F.abs("coefficient") > coef_threshold, 1).otherwise(0))
        .withColumn("zero", 1 - F.col("nonzero"))
    )

    # Aggregate stats
    agg_df = (
        prepped_df
        .groupBy("feature")
        .agg(
            F.avg("coefficient").alias("mean_coef"),
            F.stddev("coefficient").alias("std_coef"),
            F.avg("abs_coef").alias("mean_abs_coef"),
            F.min("coefficient").alias("min_coef"),
            F.max("coefficient").alias("max_coef"),
            F.sum("nonzero").alias("freq_nonzero"),
            F.sum("zero").alias("freq_zero"),
            F.count("*").alias("total_obs"),
            (F.sum("nonzero") / F.count("*")).alias("nonzero_ratio"),
            (F.sum("zero") / F.count("*")).alias("zero_ratio")
        )
    )

    # Add coefficient of variation after aggregation
    result_df = (
        agg_df
        .withColumn("coef_var", F.col("std_coef") / (F.abs(F.col("mean_coef")) + F.lit(1e-12)))
        .orderBy(F.desc("nonzero_ratio"), F.desc("mean_abs_coef"))
    )

    return result_df