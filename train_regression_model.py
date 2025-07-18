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


def train_regression_model(
    train_df: DataFrame,    
    feature_cols: list,
    label_col: str,
    best_params: Dict[str, float],
    test_df: Optional[DataFrame]=None,
    weight_col: Optional[str] = None,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: bool = True
) -> Tuple[LinearRegressionModel, DataFrame, DataFrame]:
    """
    Trains a final linear regression model with best parameters and calculates comprehensive statistics.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        label_col: Name of label column
        best_params: Dictionary of best parameters from CV
        weight_col: Optional weight column name for WLS
        model_path: Optional directory path to save model
        model_name: Optional model name to use when saving
        verbose: If True, print progress and diagnostics

    Returns:
        Tuple of (trained model, metrics Spark DataFrame, coefficients Spark DataFrame)
    """
    spark = train_df.sparkSession

    # Edge case: Check empty DataFrames
    
    # Validate train
    if train_df is None or train_df.rdd.isEmpty():
        raise ValueError("Training DataFrame is empty or None.")

    # Validate test (only if provided)
    if test_df is not None:
        if test_df.rdd.isEmpty():
            raise ValueError("Testing DataFrame is empty.")


    if not feature_cols:
        raise ValueError("No feature columns provided.")

    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    missing_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in DataFrame: {missing_features}")

    # # Initialize model
    # if verbose:
    #     print("Assembling features...")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_df = assembler.transform(train_df)

    if test_df is not None:
        test_df = assembler.transform(test_df)

     # Initialize and train model
    if weight_col:
        print("I\nTraining Weighted Linear Regression model (WLS)...")
        lr = LinearRegression(featuresCol="features", labelCol=label_col, weightCol=weight_col, **best_params)
    else:
        print("\nTraining Linear Regression model (OLS) ...")
        lr = LinearRegression(featuresCol="features", labelCol=label_col, **best_params)


    # if verbose:
    #     print("\nFitting model on training data...")
    try:
        model = lr.fit(train_df)
    except Exception as e:
        print("Model training failed.")
        raise e

      # Save model if path is provided
    if model_path and model_name:
        try:
            if verbose:
                print(f"Saving model to {model_path + model_name}")
            model.write().overwrite().save(model_path + model_name)
        except Exception as e:
            print("Warning: Model saving failed.")
            print(e)

    # Make predictions
    if verbose:
        print("Generating predictions and evaluating performance...")
    train_pred = model.transform(train_df)

    if test_df is not None:
        test_pred = model.transform(test_df)

    # Metrics setup
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")
    n_train = train_df.count()
    if test_df is not None:
        n_test = test_df.count()
    p = len(model.coefficients)

    def adjusted_r2(r2: float, n: int, p: int) -> float:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float('nan')

    # # Metrics
    # if verbose:
    #     print("Evaluating performance...")

    try:
        # Always compute train metrics
        r2_train = model.summary.r2
        train_metrics = Row(
            dataset="train",
            r2=r2_train,
            adj_r2=adjusted_r2(r2_train, n_train, p),
            rmse=model.summary.rootMeanSquaredError,
            mae=evaluator.setMetricName("mae").evaluate(train_pred),
            n_observations=n_train
        )
        
        metrics_data = [train_metrics]

        if test_df is not None:
            r2_test = evaluator.setMetricName("r2").evaluate(test_pred)
            test_metrics = Row(
                dataset="test",
                r2=r2_test,
                adj_r2=adjusted_r2(r2_test, n_test, p),
                rmse=evaluator.setMetricName("rmse").evaluate(test_pred),
                mae=evaluator.setMetricName("mae").evaluate(test_pred),
                n_observations=n_test
            )
            metrics_data.append(test_metrics)

        metrics_df = spark.createDataFrame(metrics_data)

    except Exception as e:
        raise RuntimeError(f"Failed to compute evaluation metrics: {e}")


    # Coefficients summary
    if verbose:
        print("Extracting model coefficients...")
    coefficients_data = [
        Row(feature="intercept", coefficient=float(model.intercept))
    ] + [
        Row(feature=col, coefficient=float(coeff))
        for col, coeff in zip(feature_cols, model.coefficients)
    ]
    coefficients_df = spark.createDataFrame(coefficients_data)

    # Check for all-zero or negative-only coefficients (edge diagnostic)
    if all(c.coefficient <= 0 for c in coefficients_df.collect()[1:]):
        print("Warning: All coefficients are non-positive. Consider re-evaluating your feature set or regularization strength.")

    if verbose:
        print("Model training and evaluation completed successfully.")
    if verbose:
        print( '\nMetrics summary')
        print(metrics_df.show())

    return model, metrics_df, coefficients_df
