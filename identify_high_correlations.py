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

def identify_high_correlations(corr_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Identify feature pairs with correlation above the threshold and sort by descending absolute correlation values.
    
    Args:
        corr_df (pd.DataFrame): Correlation matrix (square matrix where rows and columns are features).
        threshold (float): Minimum absolute correlation value to consider.

    Returns:
        pd.DataFrame: DataFrame containing pairs of highly correlated features.
    """
    if not isinstance(corr_df, pd.DataFrame):
        raise TypeError("Expected 'corr_df' to be a Pandas DataFrame.")

    if corr_df.isnull().values.any():
        print("Warning: Correlation matrix contains NaNs. Consider filling or dropping NaNs.")

    high_corr_pairs = []
    columns = corr_df.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):  # Avoid duplicates and self-correlation
            corr_value = corr_df.iloc[i, j]

            if pd.notna(corr_value):  # Skip NaNs
                abs_corr_value = builtins.abs(corr_value)

                if abs_corr_value > threshold:
                    high_corr_pairs.append({
                        'feature1': columns[i],
                        'feature2': columns[j],
                        'correlation': corr_value,
                        'abs_correlation': abs_corr_value
                    })

    # Convert to DataFrame
    high_corr_df = pd.DataFrame(high_corr_pairs)

    if high_corr_df.empty:
        print("Warning: No feature pairs found with correlation above the threshold.")
        return high_corr_df  # Return empty DataFrame

    return high_corr_df.sort_values(by='abs_correlation', ascending=False).reset_index(drop=True)

