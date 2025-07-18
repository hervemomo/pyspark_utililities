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


 
def run_variance_wls_pipeline(
    df_train,
    df_val,
    results,
    thresholds,
    target,
    models_path,
    data_path,
    var_param_grid,
    wls_param_grid,
    var_num_folds=3,
    wls_num_folds=3,
    max_stages=3,
    plot_diagnostics=True
):
 
    
    """
    Executes a multi-stage Weighted Least Squares (WLS) regression pipeline with iterative variance modeling 
    to address heteroscedasticity in residuals.
 
    This function:
    - Uses residuals from a base model to train a variance model predicting log-squared residuals.
    - Transforms predicted variances into weights for WLS regression.
    - Optionally repeats this process for multiple stages and thresholds.
    - Stores final variance and WLS models with their coefficients, metrics, and CV results.
 
    Args:
        df_train (DataFrame): Spark DataFrame for training.
        df_val (DataFrame): Spark DataFrame for validation/testing.
        results (dict): Dictionary containing base model references and their selected features.
        thresholds (list[float]): Variance feature selection thresholds (e.g., for LASSO).
        target (str): Name of the continuous target variable.
        models_path (str): Directory path to save models.
        data_path (str): Directory path to save intermediate artifacts.
        var_param_grid (dict): Hyperparameter grid for the variance model.
        wls_param_grid (dict): Hyperparameter grid for the WLS model.
        var_num_folds (int, optional): Number of cross-validation folds for the variance model. Default is 3.
        wls_num_folds (int, optional): Number of cross-validation folds for the WLS model. Default is 3.
        max_stages (int, optional): Number of iterative stages for alternating variance/WLS training. Default is 3.
        plot_diagnostics (bool, optional): Whether to plot residual diagnostics at each stage. Default is True.
 
    Returns:
        tuple:
            - variance_dict (dict): Contains final variance models, parameters, metrics, and CV results for each threshold.
            - wls_dict (dict): Contains final WLS models, weights, parameters, metrics, and CV results for each threshold.
 
    Notes:
        - If max_stages = 3, the pipeline runs: [base model → variance model → WLS → variance model].
        - The function assumes the presence of a callable `lasso_feature_selection_experiment` and plotting helper `plot_residual_diagnostics`.
        - Designed for use in healthcare or econometric contexts with large Spark datasets and heteroscedastic targets.
    """
 
    variance_dict = {}
    wls_dict = {}
 
    model_list = [v['model_name'] for v in results.values()]
    feature_list = [[f for f in v['feature_cols'] if f != 'intercept'] for v in results.values()]
    model_features_list = list(zip(thresholds, model_list, feature_list))
 
    for thr, base_model, feat_col in model_features_list:
        print(f"\n===== STARTING WLS ITERATION for threshold {thr} =====")
 
        for stage in range(1, max_stages + 1):
            print(f"\n--- STAGE {stage} ---")
 
            # Assemble features
            assembler = VectorAssembler(inputCols=feat_col, outputCol='features')
            train_assembled = assembler.transform(df_train).cache()
            val_assembled = assembler.transform(df_val).cache()
 
            # Predict and calculate residuals using current model
            model = base_model if stage == 1 else wls_model
            plot_df_train = (
                model.transform(train_assembled)
                .withColumn('residual', F.col(target) - F.col('prediction'))
                .withColumn('squared_residual', F.pow(F.col('residual'), 2))
                .withColumn('log_squared_residual', F.log(F.col('squared_residual')))
                .cache()
            )
 
            plot_df_val = (
                model.transform(val_assembled)
                .withColumn('residual', F.col(target) - F.col('prediction'))
                .withColumn('squared_residual', F.pow(F.col('residual'), 2))
                .withColumn('log_squared_residual', F.log(F.col('squared_residual')))
                .cache()
            )
 
            # Plot residual diagnostics
            required_cols = {'prediction', target, 'residual', 'squared_residual', 'log_squared_residual'}
            if plot_diagnostics and required_cols.issubset(set(plot_df_train.columns)):
                try:
                    hp.plot_residual_diagnostics(plot_df_train, label_col=target)
                except Exception as e:
                    print(f"Plotting failed: {e}")
 
            # Prepare for variance modeling
            train_with_residual = plot_df_train.drop('prediction', 'features')
            val_with_residual = plot_df_val.drop('prediction', 'features')
 
            # === 1. Variance Modeling ===
            print(f"Running variance modeling...")
            var_model, feat, var_bp, var_metrics, var_coeffs, var_cv_results = lasso_feature_selection_experiment(
                train_df=train_with_residual,
                test_df=val_with_residual,
                variance_df=None,
                all_features=feat_col,
                target='log_squared_residual',
                threshold=thr,
                models_path=models_path,
                data_path=data_path,
                exp_name=f"variance_thr{str(thr).replace('.', '')}_stage{stage}",
                param_grid=var_param_grid,
                num_folds=var_num_folds,
            )
 
            # If final stage: STOP HERE (skip WLS)
            if stage == max_stages:
                thr_key = str(thr).replace('.', '')
                variance_dict[thr_key] = {
                    'model_name': var_model,
                    'feature_cols': feat_col,
                    'best_params': var_bp,
                    'metrics_df': var_metrics,
                    'cv_results': var_cv_results,
                    'weights_df': None,
                    'coeff_df': var_coeffs
                }
 
                print(f"Final variance model saved for backtransformation (threshold={thr})")
                break
 
            # === 2. Generate Weights from Variance Model ===
            weights_df = (
                var_model.transform(assembler.transform(train_with_residual))
                .withColumn("predicted_variance", F.exp(F.col("prediction")))
                .withColumn("weights", 1 / F.col("predicted_variance"))
                .drop("features", "prediction")
                .cache()
            )
 
            # === 3. WLS Modeling ===
            print(f"Running WLS modeling...")
            wls_model, feat, wls_bp, wls_metrics, wls_coeffs, wls_cv_results = lasso_feature_selection_experiment(
                train_df=weights_df,
                test_df=df_val,
                variance_df=None,
                weight_col='weights',
                all_features=feat_col,
                target=target,
                threshold=thr,
                models_path=models_path,
                data_path=data_path,
                exp_name=f"wls_thr{str(thr).replace('.', '')}_stage{stage}",
                param_grid=wls_param_grid,
                num_folds=wls_num_folds,
            )
 
            # Save intermediate WLS results
            if stage == max_stages - 1:  # Save final WLS model (before last variance-only stage)
                thr_key = str(thr).replace('.', '')
                weight_audit = weights_df.select('dad_transaction_id', 'inst_code', 'fiscal_year', 'weights')
                wls_dict[thr_key] = {
                    'model_name': wls_model,
                    'feature_cols': feat_col,
                    'best_params': wls_bp,
                    'metrics_df': wls_metrics,
                    'cv_results': wls_cv_results,
                    'weights_df': weight_audit,
                    'coeff_df': wls_coeffs
                }
 
            # Cleanup cache
            train_assembled.unpersist()
            val_assembled.unpersist()
            plot_df_train.unpersist()
            plot_df_val.unpersist()
            weights_df.unpersist()
 
    print("WLS pipeline completed.")
    return variance_dict, wls_dict