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


def cross_validate_linear_regression(
    train_df: DataFrame,
    feature_cols: list,
    label_col: str,
    param_grid: Dict[str, List[float]],
    num_folds: int = 3,
    weight_col: Optional[str] = None,
    parallelism: int = 2,  # safer default
    sample_fraction: Optional[float] = None,
    repartition_n: Optional[int] = None,
    verbose: bool = True
) -> Tuple[LinearRegressionModel, DataFrame]:
    """
    Perform k-fold cross-validation for Linear Regression (OLS/WLS) safely.
    Optionally supports sampling and repartitioning for memory control.
    """

    # Input validation
    if not isinstance(train_df, DataFrame):
        raise TypeError("train_df must be a Spark DataFrame")
    if not feature_cols:
        raise ValueError("feature_cols list cannot be empty")
    missing = set(feature_cols + [label_col]) - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
    if weight_col and weight_col not in train_df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found")
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if not param_grid or any(len(v) == 0 for v in param_grid.values()):
        raise ValueError("param_grid must specify at least one parameter with values")

    if sample_fraction:
        if verbose:
            print(f"[INFO] Sampling {sample_fraction*100:.1f}% of data for CV...")
        train_df = train_df.sample(withReplacement=False, fraction=sample_fraction, seed=42)

    if repartition_n:
        if verbose:
            print(f"[INFO] Repartitioning training data to {repartition_n} partitions...")
        train_df = train_df.repartition(repartition_n)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    # data = assembler.transform(train_df).select('features', label_col, *(weight_col if weight_col else []))

    # Added this step to minimize memory usage
    if weight_col is not None:
        data = assembler.transform(train_df).select('features', label_col, weight_col)
    else:
        data = assembler.transform(train_df).select('features', label_col)

    data.persist(StorageLevel.MEMORY_AND_DISK)
    number_of_rows = data.count()  # trigger caching

    if verbose:
        start_time = time.time()
        print(f"- Number of rows: {number_of_rows}")
        print(f"- Number of features: {len(feature_cols)}")
        print(f"- Label column: '{label_col}'")
        if weight_col: print(f"- Weight column: '{weight_col}'")
        print(f"- Folds: {num_folds}")
        print(f"- Parallelism: {parallelism}")
        print(f"\nParameter grid to evaluate:")
        for param, values in param_grid.items():
            print(f"- {param}: {values}")

    lr = LinearRegression(labelCol=label_col, featuresCol='features')
    if weight_col: lr = lr.setWeightCol(weight_col)

    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='rmse')
    grid = ParamGridBuilder()
    for param, values in param_grid.items():
        grid = grid.addGrid(lr.getParam(param), values)
    param_grid_built = grid.build()

    if verbose:
        print(f"- Total parameter combinations: {len(param_grid_built)}")
        print("Starting cross-validation...")

    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=param_grid_built,
        evaluator=evaluator,
        numFolds=num_folds,
        parallelism=parallelism,
        seed=42
    )

    try:
        fit_start = time.time()
        cv_model = cv.fit(data)
        fit_time = time.time() - fit_start
        if verbose:
            print(f"[INFO] CV completed in {fit_time:.2f} seconds.")
    except Py4JJavaError as e:
        msg = e.java_exception.getMessage()
        raise RuntimeError(f"[FATAL] CV failed: {msg}")

    best_model = cv_model.bestModel
    avg_metrics = cv_model.avgMetrics
    min_rmse = builtins.min(avg_metrics)
    param_maps = cv_model.getEstimatorParamMaps()

    results = []
    for i, (pm, rmse) in enumerate(zip(param_maps, avg_metrics), 1):
        rec = {
            "experiment_num": i,
            "avg_rmse": float(rmse),
            "is_best": float(rmse) == float(min_rmse),
            "weight_used": bool(weight_col)
        }
        for param in param_grid:
            val = pm.get(lr.getParam(param))
            rec[param] = float(val) if isinstance(val, (float, int)) else val
        results.append(rec)

    spark = train_df.sparkSession
    results_df = spark.createDataFrame(results)

    if verbose:
        total_time = time.time() - start_time
        print("\n=== Cross-Validation Summary ===")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best RMSE: {min_rmse:.4f}")
        best_row = results_df.filter(F.col("is_best") == True).first()
        for param in param_grid:
            print(f"- {param}: {best_row[param]}")
        print("="*60)

    data.unpersist()
    return best_model, results_df