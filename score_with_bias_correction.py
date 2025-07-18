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


 
def score_with_bias_correction(
    df: DataFrame,
    feature_cols: list,
    wls_model_path: str,
    variance_model_path: Optional[str] = None,
    output_col: str = "back_transformed_cost"
) -> DataFrame:
    """
    Score a new DataFrame using a pre-trained WLS model and optionally a variance model.
    Applies exponential bias correction if variance_model_path is provided.
 
    Parameters:
    - df: Input DataFrame with feature columns
    - feature_cols: List of feature column names used during training
    - wls_model_path: Path to the saved WLS model
    - variance_model_path: (Optional) Path to the saved variance model    
    - output_col: Name of the column to store the back-transformed cost (default: back_transformed_cost)
 
    Returns:
    - DataFrame with additional columns: predicted log-cost, (optional) predicted variance, back-transformed output
    """
 
    print("Starting model scoring...")
 
    # 1. Assemble features
    print("🔧 Assembling features...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_features = assembler.transform(df)
 
    # 2. Load WLS model
    print("Loading WLS model...")
    wls_model = LinearRegressionModel.load(wls_model_path)
 
    # Validate number of features
    num_model_features = len(wls_model.coefficients)
    if len(feature_cols) != num_model_features:
        raise ValueError(
            f"Mismatch between number of input features ({len(feature_cols)}) "
            f"and number of model coefficients ({num_model_features}). "
            "Check that feature_cols matches training features used for the model."
        )
 
    wls_model.setPredictionCol("wls_lgcost_prediction")
    df_pred = wls_model.transform(df_features)
    target_prediction = wls_model.getPredictionCol()
    print("WLS prediction column:", target_prediction)
 
    if variance_model_path is not None:
        # 3. Load and apply variance model
        print("Predicting residual variance...")
        variance_model = LinearRegressionModel.load(variance_model_path)
        variance_model.setPredictionCol("variance_prediction")
        df_var = variance_model.transform(df_pred)
 
        variance_col = "predicted_variance"
        df_var = df_var.withColumn(variance_col, F.exp(F.col("variance_prediction")))
 
        # 4. Apply bias correction
        print("Applying exponential bias correction with variance...")
        df_corrected = df_var.withColumn(
            output_col,
            F.exp(F.col(target_prediction) + (F.col(variance_col) / 2))
        )
 
        return df_corrected.select(*df.columns, target_prediction, variance_col, output_col)
    else:
        # No variance model: simple exponential back-transform
        print("No variance model provided. Skipping bias correction...")
        df_corrected = df_pred.withColumn(
            output_col,
            F.exp(F.col(target_prediction))
        )
        print('Scoring completed')
 
        return df_corrected.select(*df.columns, target_prediction, output_col)