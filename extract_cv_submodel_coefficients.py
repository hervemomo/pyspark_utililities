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



def extract_cv_submodel_coefficients(cv_model, feature_cols):
    """
    Extracts submodel coefficients and intercepts from a CrossValidatorModel with collectSubModels=True.
    Outputs a flattened DataFrame with one row per (fold, grid, feature), and each hyperparameter in its own column.

    Args:
        cv_model: Trained CrossValidatorModel (with collectSubModels=True)
        feature_cols: List of feature names used in the model

    Returns:
        Spark DataFrame with columns:
        fold_id, feature, coefficient, intercept, <param1>, <param2>, ...
    """
    spark = SparkSession.builder.getOrCreate()
    param_grid = cv_model.getEstimatorParamMaps()
    num_folds = len(cv_model.subModels)

    all_rows = []
    for fold_idx, fold_models in enumerate(cv_model.subModels):
        for param_idx, model in enumerate(fold_models):
            param_map = param_grid[param_idx]
            coefficients = model.coefficients.toArray()
            intercept = model.intercept

            # Flatten param map
            param_values = {p.name: param_map[p] for p in param_map}

            for feat_name, coef in zip(feature_cols, coefficients):
                row = {
                    "fold_id": fold_idx,
                    "feature": feat_name,
                    "coefficient": float(coef),
                    "intercept": float(intercept)
                }
                row.update(param_values)
                all_rows.append(row)

    # Convert list of dicts to DataFrame, infer schema
    return spark.createDataFrame(all_rows)
