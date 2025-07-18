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

import logging
from functools import wraps

from pyspark.storagelevel import StorageLevel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import *
from typing import Dict, List, Optional, Tuple # for hint type
from pyspark.ml.regression import LinearRegression, LinearRegressionModel


def get_best_params_from_cv(cv_results_df: DataFrame, verbose: bool = False) -> Dict[str, float]:
    """
    Extracts best hyperparameter settings from cross-validation results DataFrame.

    Args:
        cv_results_df: Cross-validation results DataFrame containing an 'is_best' flag column.
        verbose: If True, print selected best parameters.

    Returns:
        Dictionary of best hyperparameters.
    
    Raises:
        ValueError if no best row is found.
    """
    # Try both True and 1 to be safe
    best_row = cv_results_df.filter("is_best = True OR is_best = 1").first()

    if best_row is None:
        raise ValueError("No best model found in the CV results (no row where is_best = True).")

    excluded_keys = {"experiment_num", "avg_rmse", "is_best", "weight_used"}
    best_params = {k: v for k, v in best_row.asDict().items() if k not in excluded_keys}

    if verbose:
        print("Best parameters extracted:", best_params)

    return best_params