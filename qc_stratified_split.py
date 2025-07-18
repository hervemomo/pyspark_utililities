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


from pyspark.sql.types import *
from typing import Dict, List, Optional, Tuple # for hint type




def qc_stratified_split(
    df: DataFrame,
    train_df: DataFrame,
    test_df: DataFrame,
    strata_col: str,
    desired_frac: float = 0.8
) -> Tuple[DataFrame, DataFrame]:
    """
    Compute QC metrics for a stratified train/test split.

    Parameters
    ----------
    df : DataFrame
        The original full DataFrame before splitting.
    train_df : DataFrame
        The training set returned by your split function.
    test_df : DataFrame
        The test set returned by your split function.
    strata_col : str
        Name of the categorical column used for stratification.
    desired_frac : float, default=0.8
        The target fraction in the training set (0 < desired_frac < 1).

    Returns
    -------
    stats_df : DataFrame
        Per-stratum QC table with columns:
        [strata_col, n_total, n_train, n_test, 
         p_train, p_test, diff]
    summary_df : DataFrame
        Single-row summary with columns:
        [overall_train_frac, min_dev, median_dev, max_dev]
    """
    # 1) Count per stratum in full/train/test
    total_counts = df.groupBy(strata_col)\
                          .agg(F.count("*").alias("n_total"))
    train_counts = train_df.groupBy(strata_col)\
                           .agg(F.count("*").alias("n_train"))
    test_counts  = test_df.groupBy(strata_col)\
                          .agg(F.count("*").alias("n_test"))

    # 2) Join counts into one DataFrame
    stats_df = (
        total_counts
        .join(train_counts, strata_col, "left")
        .join(test_counts,  strata_col, "left")
        .fillna(0)  # handle strata missing in train or test
    )

    # 3) Compute proportions and absolute deviation
    stats_df = (
        stats_df
        .withColumn("p_train", F.col("n_train") / F.col("n_total"))
        .withColumn("p_test",  F.col("n_test")  / F.col("n_total"))
        .withColumn("diff",     F.abs(F.col("p_train") - F.lit(desired_frac)))
    )

    # 4) Compute overall train fraction
    print("Overall split ratio: ")
    overall_frac = train_df.count() / df.count()

    print(f"  train / full  = {overall_frac:.4f}  (target = {desired_frac:.2f})\n")

    # 5) Build a one-row summary DataFrame
    summary_df = (
        stats_df.agg(
            F.min("diff").alias("min_dev"),
            F.expr("percentile_approx(diff, 0.5)").alias("median_dev"),
            F.max("diff").alias("max_dev")
        )
        .withColumn("overall_train_frac", F.lit(overall_frac))
        # Reorder columns
        .select("overall_train_frac", "min_dev", "median_dev", "max_dev")
    )

    return stats_df, summary_df