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

def plot_residuals(df, pred_col="pred_total_cost", resid_col="residual"):
    """
    Plots Residuals vs. Predicted Values (Scatter Plot) and Residual Boxplot.

    Parameters:
    df (pyspark.sql.DataFrame): PySpark DataFrame containing predicted values and residuals.
    pred_col (str): Column name for predicted values.
    resid_col (str): Column name for residuals.
    
    Returns:
    None
    """
    # Convert to Pandas for visualization
    residuals_pd = df.select(pred_col, resid_col).toPandas()

    # Create a 2x1 figure for Residual Histogram and Boxplot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Plot 1: Residual vs. Predicted Values (Scatter Plot)
    axs[0].scatter(
        residuals_pd[pred_col], residuals_pd[resid_col], alpha=0.7, edgecolors="black"
    )
    axs[0].axhline(y=0, color="red", linestyle="--", linewidth=1)  # Reference line at zero
    axs[0].set_title("Residuals vs. Predicted Values")
    axs[0].set_xlabel("Predicted Values")
    axs[0].set_ylabel("Residuals")

    # Plot 2: Boxplot of Residuals
    axs[1].boxplot(
        residuals_pd[resid_col], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue")
    )
    axs[1].set_title("Boxplot of Residuals")
    axs[1].set_xlabel("Residual")

    plt.tight_layout()
    plt.show();  # Suppresses any unwanted return values