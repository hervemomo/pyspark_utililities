# PySpark Helper Functions
Author: Herve Momo


A  collection of reusable PySpark utility functions to accelerate PySpark experimentation with consistent, documented, and team-friendly practices. These helpers cover preprocessing, modeling, diagnostics, and IO operations with a focus on clean code and collaboration. 

# 📚 Table of Contents

## 1. Data Preprocessing & Feature Engineering
- [`one_hot_encode_spark`](#one_hot_encode_spark)
- [`add_interaction_columns`](#add_interaction_columns)
- [`compute_feature_variance`](#compute_feature_variance)

## 2. Train/Test Splitting
- [`stratified_train_test_split`](#stratified_train_test_split)
- [`qc_stratified_split`](#qc_stratified_split)

## 3. EDA & Statistical Analysis
- [`calculate_correlation_matrix`](#calculate_correlation_matrix)
- [`identify_high_correlations`](#identify_high_correlations)
- [`summary_stats`](#summary_stats)
- [`get_shape`](#get_shape)
- [`get_column_types`](#get_column_types)

## 4. Modeling & Cross-Validation
- [`train_regression_model`](#train_regression_model)
- [`cross_validate_linear_regression`](#cross_validate_linear_regression)
- [`get_best_params_from_cv`](#get_best_params_from_cv)
- [`cross_validate_feature_selection`](#cross_validate_feature_selection)
- [`extract_cv_submodel_coefficients`](#extract_cv_submodel_coefficients)
- [`compute_cv_feature_importance`](#compute_cv_feature_importance)
- [`run_variance_wls_pipeline`](#run_variance_wls_pipeline)

## 5. Model Evaluation & Plotting
- [`score_with_bias_correction`](#score_with_bias_correction)
- [`compute_regression_metrics`](#compute_regression_metrics)
- [`plot_residual_diagnostics`](#plot_residual_diagnostics)
- [`plot_cv_metric_by_experiment`](#plot_cv_metric_by_experiment)

## 6. DataFrame Operations
- [`join_dataframes`](#join_dataframes)
- [`compare_dataframes`](#compare_dataframes)

## 7. Saving & Loading Data
- [`save_to_excel`](#save_to_excel)
- [`save_to_parquet`](#save_to_parquet)
- [`load_ordered_parquet`](#load_ordered_parquet)

## 8. System Utilities & Logging
- [`time_logger`](#time_logger)
- [`display_folder_info`](#display_folder_info)

---
## 📦 Data Preprocessing & Feature Engineering

### one_hot_encode_spark
**Description**  
Performs one-hot encoding on multiple categorical variables in a PySpark DataFrame. The function handles missing or invalid categories, ensures sparse vectors are converted to arrays, and generates new binary columns with clear naming. It safely returns the original DataFrame if no valid encoding columns are specified.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input DataFrame containing categorical columns to encode.

`to_encode_cols` (`list`)  
List of categorical column names to one-hot encode. Columns not found in the DataFrame are ignored.

**Returns**  
`df_final` (`pyspark.sql.DataFrame`)  
A transformed DataFrame that includes the original unencoded columns, original categorical columns, and one-hot encoded binary columns for each category level.

**Example**  
```python
from helpers import one_hot_encode_spark

# Sample PySpark DataFrame with a categorical column 'color'
df = spark.createDataFrame([
    ("red",), ("blue",), ("green",), ("blue",)
], ["color"])

# Apply one-hot encoding
encoded_df = one_hot_encode_spark(df, ["color"])

encoded_df.show()
```
### add_interaction_columns
**Description**  
Generates interaction terms between pairs of columns in a PySpark DataFrame, based on definitions provided in a dictionary. The function efficiently creates all specified interactions in a single transformation. It safely skips any column pairs if either column is missing in the DataFrame.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
Input DataFrame containing the features to interact.

`interaction_definitions` (`dict`)  
Dictionary specifying interaction definitions.  
Each key is a group name, and the value is a tuple of two lists:  
- The first list contains column names to use as the first operand.  
- The second list contains column names to use as the second operand.  
Each pair `(col1, col2)` produces a new column named `col1_col2`.

**Returns**  
`df` (`pyspark.sql.DataFrame`)  
DataFrame with all original columns and newly generated interaction columns.

**Example**  
```python
from helpers import add_interaction_columns

# Define interaction definitions
interactions = {
    "group1": (["age", "income"], ["bmi", "smoker"])
}

# Apply function
df_with_interactions = add_interaction_columns(df, interactions)

df_with_interactions.show()
```
### compute_feature_variance
**Description**  
Computes the variance of binary (0/1) features in a PySpark DataFrame. It returns the count of 1s and 0s, their proportions, and the variance for each specified feature. Optionally filters features below a minimum variance threshold. Handles edge cases like empty DataFrames, invalid column names, and out-of-range thresholds.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input DataFrame containing binary features.

`feature_cols` (`list`)  
List of binary feature column names to analyze.

`variance_threshold` (`float`, optional)  
Minimum variance required to retain a feature. If `None`, no filtering is applied. Valid range is [0, 0.25].

**Returns**  
`narrow_df` (`pyspark.sql.DataFrame`)  
A DataFrame containing the following columns for each input feature:  
- `feature`: Name of the binary feature  
- `count_of_1`: Count of rows where the feature is 1  
- `count_of_0`: Count of rows where the feature is 0  
- `proportion_of_1`: Proportion of 1s  
- `proportion_of_0`: Proportion of 0s  
- `variance`: Computed variance of the feature

**Example**  
```python
from helpers import compute_feature_variance

# Example usage on a binary-feature DataFrame
binary_features = ["feature_A", "feature_B", "feature_C"]
result_df = compute_feature_variance(df, binary_features, variance_threshold=0.01)

result_df.show()
```

## 📦 Train/Test Splitting

### stratified_train_test_split
**Description**  
Performs a stratified train/test split on a PySpark DataFrame. Ensures that each level of a specified categorical column is represented in the training set at a specified fraction. This helps preserve the distribution of categorical groups across both datasets.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input DataFrame to split.

`strata_col` (`str`)  
Name of the categorical column to stratify on.

`train_frac` (`float`, default=`0.8`)  
Fraction of each stratum to include in the training set (must be between 0 and 1).

`seed` (`int`, default=`42`)  
Random seed for reproducibility.

`uid_col` (`str`, default=`"_strat_uid"`)  
Temporary column name for generating a unique identifier. This column is dropped before the result is returned.

**Returns**  
`train_df` (`pyspark.sql.DataFrame`)  
The stratified training set.

`test_df` (`pyspark.sql.DataFrame`)  
The stratified test set containing the remaining rows.

**Example**  
```python
from helpers import stratified_train_test_split

# Suppose `df` has a column "region" with values like "North", "South", "East", "West"
train_df, test_df = stratified_train_test_split(
    df=df,
    strata_col="region",
    train_frac=0.75,
    seed=123
)

print("Train count:", train_df.count())
print("Test count:", test_df.count())
```

### qc_stratified_split 
**Description**  
Computes quality control (QC) metrics for a stratified train/test split. Evaluates whether the proportions of each stratum in the training and test sets match the desired split fraction. Returns both per-stratum statistics and an overall summary of deviations.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The original full DataFrame before splitting.

`train_df` (`pyspark.sql.DataFrame`)  
The training set generated from the split.

`test_df` (`pyspark.sql.DataFrame`)  
The test set generated from the split.

`strata_col` (`str`)  
Name of the categorical column used for stratification.

`desired_frac` (`float`, default=`0.8`)  
The expected fraction of each stratum that should appear in the training set.

**Returns**  
`stats_df` (`pyspark.sql.DataFrame`)  
Per-stratum QC table with the following columns:  
- `strata_col`: Stratum level  
- `n_total`: Total count in the full dataset  
- `n_train`: Count in training set  
- `n_test`: Count in test set  
- `p_train`: Proportion of training samples in stratum  
- `p_test`: Proportion of test samples in stratum  
- `diff`: Absolute deviation from the expected training fraction

`summary_df` (`pyspark.sql.DataFrame`)  
A one-row summary with the following columns:  
- `overall_train_frac`: Actual overall train fraction  
- `min_dev`: Minimum deviation across strata  
- `median_dev`: Median deviation across strata  
- `max_dev`: Maximum deviation across strata

**Example**  
```python
from helpers import qc_stratified_split

# Compute QC metrics after stratified splitting
stats_df, summary_df = qc_stratified_split(
    df=df,
    train_df=train_df,
    test_df=test_df,
    strata_col="region",
    desired_frac=0.8
)

stats_df.show()
summary_df.show()
```


## 📦 EDA & Statistical Analysis

### calculate_correlation_matrix
**Description**  
Calculates the Pearson correlation matrix for a list of numeric features in a PySpark DataFrame and returns it as a Pandas DataFrame. This allows for further inspection or visualization outside of Spark. It uses `VectorAssembler` and Spark’s `Correlation` module.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
Input DataFrame containing numerical features.

`feature_cols` (`list`)  
List of numerical column names to compute pairwise correlations on.

**Returns**  
`corr_df` (`pandas.DataFrame`)  
A square Pandas DataFrame where rows and columns correspond to features and values represent Pearson correlation coefficients.

**Example**  
```python
from helpers import calculate_correlation_matrix

# List of numeric features
features = ["age", "income", "bmi"]

# Compute correlation matrix
corr_df = calculate_correlation_matrix(df, features)

print(corr_df)
```
### identify_high_correlations  
**Description**  
Identifies pairs of features in a correlation matrix that have an absolute correlation above a specified threshold. Skips NaN values and ensures duplicate and self-pairs are excluded. Returns the results sorted by descending absolute correlation.

**Parameters**  
`corr_df` (`pandas.DataFrame`)  
A square Pandas correlation matrix with identical row and column names representing feature names.

`threshold` (`float`, default=`0.0`)  
Minimum absolute correlation value to consider for reporting feature pairs.

**Returns**  
`high_corr_df` (`pandas.DataFrame`)  
A Pandas DataFrame containing highly correlated feature pairs with the following columns:  
- `feature1`: First feature in the pair  
- `feature2`: Second feature in the pair  
- `correlation`: Actual correlation value  
- `abs_correlation`: Absolute value of the correlation

**Example**  
```python
from helpers import identify_high_correlations

# Assume corr_df is a correlation matrix computed earlier
high_corr_df = identify_high_correlations(corr_df, threshold=0.8)

print(high_corr_df.head())
```
### summary_stats 
**Description**  
Computes descriptive statistics (count, mean, stddev, min, max) for all numeric columns in a PySpark DataFrame. Results are returned as a Pandas DataFrame for convenience in downstream reporting or profiling.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input DataFrame to summarize.

**Returns**  
`summary_df` (`pandas.DataFrame`)  
A Pandas DataFrame containing basic summary statistics for each numeric column.

**Example**  
```python
from helpers import summary_stats

# Get descriptive statistics
summary_df = summary_stats(df)

print(summary_df)
```
### get_shape 
**Description**  
Returns the number of rows and columns in a PySpark DataFrame. Mimics the `.shape` attribute in Pandas for convenience in exploratory data analysis.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input PySpark DataFrame.

**Returns**  
`shape` (`tuple`)  
A tuple of `(num_rows, num_columns)` representing the dimensions of the DataFrame.

**Example**  
```python
from helpers import get_shape

# Get shape of a DataFrame
num_rows, num_cols = get_shape(df)

print(f"Rows: {num_rows}, Columns: {num_cols}")
```
### get_column_types
**Description**  
Retrieves and summarizes the data types of columns in a PySpark DataFrame. Returns a Pandas DataFrame with the column names and corresponding Spark data types for easy inspection or documentation.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input DataFrame whose schema is to be examined.

**Returns**  
`types_df` (`pandas.DataFrame`)  
A Pandas DataFrame with two columns:  
- `column`: Name of the column  
- `dtype`: Spark data type of the column

**Example**  
```python
from helpers import get_column_types

# Get column types of a PySpark DataFrame
types_df = get_column_types(df)

print(types_df.head())
```

## 📦 Modeling & Cross-Validation

### train_regression_model 
**Description**  
Trains a linear regression model (OLS or WLS) using the best hyperparameters from prior cross-validation. Optionally evaluates on a test set and saves the model. Returns the trained model along with evaluation metrics and model coefficients. Includes safeguards for empty DataFrames and missing feature columns.

**Parameters**  
`train_df` (`pyspark.sql.DataFrame`)  
Training dataset used to fit the model.

`feature_cols` (`list`)  
List of feature column names to use in model training.

`label_col` (`str`)  
Name of the target (label) column.

`best_params` (`dict`)  
Best hyperparameter values selected from cross-validation (e.g., `{"regParam": 0.01, "elasticNetParam": 0.5}`).

`test_df` (`pyspark.sql.DataFrame`, optional)  
Optional test dataset for model evaluation.

`weight_col` (`str`, optional)  
Optional name of column to use for WLS weights. If omitted, performs OLS.

`model_path` (`str`, optional)  
Optional path to save the trained model.

`model_name` (`str`, optional)  
Optional model name used when saving to disk.

`verbose` (`bool`, default=`True`)  
Whether to print detailed output and progress messages.

**Returns**  
`model` (`pyspark.ml.regression.LinearRegressionModel`)  
The fitted PySpark linear regression model.

`metrics_df` (`pyspark.sql.DataFrame`)  
Evaluation metrics for training (and test, if provided). Includes R², adjusted R², RMSE, MAE, and observation count.

`coefficients_df` (`pyspark.sql.DataFrame`)  
Model coefficients, including intercept and feature-specific coefficients.

**Example**  
```python
from helpers import train_regression_model

# Train final model using best hyperparameters
model, metrics_df, coeffs_df = train_regression_model(
    train_df=train_data,
    feature_cols=["age", "bmi", "income"],
    label_col="lgcost",
    best_params={"regParam": 0.01, "elasticNetParam": 0.5},
    test_df=test_data,
    weight_col="weights",
    model_path="./models/",
    model_name="final_wls_model"
)

metrics_df.show()
coeffs_df.show()
```
### cross_validate_linear_regression 
**Description**  
Performs k-fold cross-validation on a linear regression model in PySpark. Supports both ordinary least squares (OLS) and weighted least squares (WLS) via an optional weight column. Optionally allows sampling and repartitioning for memory-efficient training. Returns the best model and a summary DataFrame of RMSE for each parameter combination.

**Parameters**  
`train_df` (`pyspark.sql.DataFrame`)  
Training DataFrame used for cross-validation.

`feature_cols` (`list`)  
List of column names to use as input features.

`label_col` (`str`)  
Name of the target column.

`param_grid` (`dict`)  
Dictionary defining hyperparameters to tune. Keys are parameter names (e.g., `"regParam"`) and values are lists of values to evaluate.

`num_folds` (`int`, default=`3`)  
Number of folds for cross-validation (must be ≥ 2).

`weight_col` (`str`, optional)  
Optional column name specifying observation weights for WLS. If not provided, defaults to OLS.

`parallelism` (`int`, default=`2`)  
Number of folds or models to evaluate in parallel.

`sample_fraction` (`float`, optional)  
Optional fraction to subsample the training data. Useful for large datasets.

`repartition_n` (`int`, optional)  
Optional number of partitions to repartition the training data before modeling.

`verbose` (`bool`, default=`True`)  
Whether to print cross-validation progress and summary metrics.

**Returns**  
`best_model` (`pyspark.ml.regression.LinearRegressionModel`)  
Trained linear regression model with the lowest average RMSE.

`results_df` (`pyspark.sql.DataFrame`)  
Summary of cross-validation results for each parameter setting, including RMSE and best model indicator.

**Example**  
```python
from helpers import cross_validate_linear_regression

# Define a basic parameter grid
param_grid = {
    "regParam": [0.01, 0.1],
    "elasticNetParam": [0.0, 0.5]
}

# Run CV with training data
best_model, cv_results = cross_validate_linear_regression(
    train_df=df,
    feature_cols=["age", "bmi", "income"],
    label_col="cost",
    param_grid=param_grid,
    num_folds=5
)

cv_results.show()
```

### get_best_params_from_cv
**Description**  
Extracts the best hyperparameter combination from a cross-validation results DataFrame. Assumes the DataFrame includes an `is_best` flag indicating the best model row. Returns a dictionary of selected parameter names and values.

**Parameters**  
`cv_results_df` (`pyspark.sql.DataFrame`)  
The DataFrame returned from cross-validation, containing hyperparameters, RMSE scores, and a boolean `is_best` column.

`verbose` (`bool`, default=`False`)  
Whether to print the selected best parameters to the console.

**Returns**  
`best_params` (`dict`)  
Dictionary mapping parameter names to their best-tuned values. Excludes metadata columns like `avg_rmse`, `is_best`, and `experiment_num`.

**Example**  
```python
from helpers import get_best_params_from_cv

# Extract best parameters from cross-validation results
best_params = get_best_params_from_cv(cv_results_df, verbose=True)

print(best_params)
```
### cross_validate_feature_selection
**Description**  
Performs k-fold cross-validation using PySpark’s `LinearRegression` for the purpose of feature selection. Accepts optional sampling and repartitioning for scalable model training. Returns the full cross-validator model including all submodels, allowing downstream extraction of coefficients for feature importance analysis.

**Parameters**  
`train_df` (`pyspark.sql.DataFrame`)  
Training DataFrame containing features and target variable.

`feature_cols` (`list`)  
List of feature column names to use in modeling.

`label_col` (`str`)  
Name of the target variable column.

`param_grid` (`dict`)  
Dictionary of hyperparameters to tune (e.g., `{"regParam": [0.01, 0.1], "elasticNetParam": [0.0, 1.0]}`).

`num_folds` (`int`, default=`3`)  
Number of cross-validation folds (must be ≥ 2).

`weight_col` (`str`, optional)  
Name of the column containing observation weights for WLS.

`parallelism` (`int`, default=`2`)  
Number of parallel tasks during cross-validation.

`sample_fraction` (`float`, optional)  
Optional sampling fraction (between 0 and 1) to reduce dataset size during CV.

`repartition_n` (`int`, optional)  
Number of partitions to repartition training data before model fitting.

`verbose` (`bool`, default=`True`)  
Whether to print diagnostic messages during processing.

**Returns**  
`cv_model` (`pyspark.ml.tuning.CrossValidatorModel`)  
The trained cross-validation model including all submodels for each fold and parameter set.

**Example**  
```python
from helpers import cross_validate_feature_selection

# Set up parameter grid
param_grid = {
    "regParam": [0.01, 0.1],
    "elasticNetParam": [0.0, 0.5, 1.0]
}

# Perform feature selection via CV
cv_model = cross_validate_feature_selection(
    train_df=train_data,
    feature_cols=["age", "bmi", "income"],
    label_col="cost",
    param_grid=param_grid,
    num_folds=5,
    weight_col=None,
    sample_fraction=0.5
)
```
### extract_cv_submodel_coefficients
**Description**  
Extracts coefficients and intercepts from each submodel in a `CrossValidatorModel` trained with `collectSubModels=True`. Returns a flattened Spark DataFrame where each row corresponds to a unique (fold, grid, feature) combination. Also includes associated hyperparameter values for each submodel.

**Parameters**  
`cv_model` (`pyspark.ml.tuning.CrossValidatorModel`)  
CrossValidatorModel returned from a feature selection or regression CV run with `collectSubModels=True`.

`feature_cols` (`list`)  
List of feature names used in the model, in the same order as they were assembled during training.

**Returns**  
`spark.sql.DataFrame`  
A long-format DataFrame containing:  
- `fold_id`: Index of the CV fold  
- `feature`: Name of the feature  
- `coefficient`: Coefficient value for the feature  
- `intercept`: Model intercept  
- `<param1>, <param2>, ...`: Each hyperparameter value (e.g., `regParam`, `elasticNetParam`)

**Example**  
```python
from helpers import extract_cv_submodel_coefficients

# Extract submodel coefficients
coef_df = extract_cv_submodel_coefficients(cv_model, feature_cols=["age", "income", "bmi"])

coef_df.show()
```
### compute_cv_feature_importance
**Description**  
Summarizes feature importance from a long-format DataFrame of cross-validated model coefficients. Computes statistical metrics such as mean, standard deviation, coefficient of variation, and frequency of non-zero coefficients to assess feature stability and relevance. Designed for use after extracting submodel coefficients from a `CrossValidatorModel`.

**Parameters**  
`cv_df` (`pyspark.sql.DataFrame`)  
DataFrame created by `extract_cv_submodel_coefficients`, containing `feature`, `coefficient`, and optionally other hyperparameters.

`coef_threshold` (`float`, default=`1e-6`)  
Threshold below which coefficients are considered zero (for non-zero frequency computation).

**Returns**  
`spark.sql.DataFrame`  
A summary DataFrame containing:  
- `feature`: Feature name  
- `mean_coef`: Mean of the coefficients  
- `std_coef`: Standard deviation of coefficients  
- `coef_var`: Coefficient of variation (std / abs(mean))  
- `mean_abs_coef`: Mean of absolute coefficients  
- `min_coef`, `max_coef`: Min/max coefficient values  
- `freq_nonzero`, `freq_zero`: Count of non-zero and zero appearances  
- `total_obs`: Total number of folds/grid evaluations  
- `nonzero_ratio`, `zero_ratio`: Proportions of non-zero/zero coefficients

**Example**  
```python
from helpers import compute_cv_feature_importance

# Compute importance summary from CV coefficient table
importance_df = compute_cv_feature_importance(coef_df)

importance_df.show()
```

### run_variance_wls_pipeline
**Description**  
Executes a multi-stage Weighted Least Squares (WLS) regression pipeline that addresses heteroscedasticity by modeling the variance of residuals. It alternates between training a variance model (on log-squared residuals) and fitting a WLS model using inverse predicted variance as weights. Intermediate results such as metrics, coefficients, and weights can be stored at each stage. Supports feature selection, cross-validation, and optional plotting.

**Parameters**  
`df_train` (`pyspark.sql.DataFrame`)  
Training DataFrame.

`df_val` (`pyspark.sql.DataFrame`)  
Validation or test DataFrame.

`results` (`dict`)  
Dictionary of base models and their selected features (from a previous modeling step).

`thresholds` (`list`)  
List of feature selection thresholds (e.g., for LASSO-based variance model feature filtering).

`target` (`str`)  
Name of the continuous target variable (e.g., `"lgcost"`).

`models_path` (`str`)  
Directory where trained models will be saved.

`data_path` (`str`)  
Directory for saving intermediate artifacts.

`var_param_grid` (`dict`)  
Hyperparameter grid for the variance model.

`wls_param_grid` (`dict`)  
Hyperparameter grid for the WLS model.

`var_num_folds` (`int`, default=`3`)  
Number of folds for cross-validating the variance model.

`wls_num_folds` (`int`, default=`3`)  
Number of folds for cross-validating the WLS model.

`max_stages` (`int`, default=`3`)  
Maximum number of pipeline stages (e.g., base model → variance model → WLS → variance model).

`plot_diagnostics` (`bool`, default=`True`)  
Whether to display residual diagnostic plots at each stage.

**Returns**  
`variance_dict` (`dict`)  
Dictionary of final variance model artifacts for each threshold:  
- `model_name`, `feature_cols`, `best_params`, `metrics_df`, `cv_results`, `coeff_df`.

`wls_dict` (`dict`)  
Dictionary of final WLS model artifacts for each threshold:  
- `model_name`, `feature_cols`, `best_params`, `metrics_df`, `cv_results`, `weights_df`, `coeff_df`.

**Example**  
```python
from helpers import run_variance_wls_pipeline

variance_results, wls_results = run_variance_wls_pipeline(
    df_train=train_df,
    df_val=val_df,
    results=base_results,
    thresholds=[0.001, 0.005],
    target="lgcost",
    models_path="./models/",
    data_path="./data/",
    var_param_grid={"regParam": [0.01], "elasticNetParam": [0.0, 0.5], "maxIter": [100]},
    wls_param_grid={"regParam": [0.01], "elasticNetParam": [0.0, 0.5], "maxIter": [100]},
    max_stages=3
)
```

## 📦 Model Evaluation & Plotting

### score_with_bias_correction
**Description**  
Scores a new DataFrame using a pre-trained Weighted Least Squares (WLS) regression model and optionally a variance model. If the variance model is provided, it applies exponential bias correction using the predicted variance to produce an unbiased estimate of the target. Ensures that feature column lengths match model expectations.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
Input DataFrame to score.

`feature_cols` (`list`)  
List of feature column names that must match the model’s training features.

`wls_model_path` (`str`)  
Path to the saved WLS model.

`variance_model_path` (`str`, optional)  
Optional path to the saved variance model used for bias correction.

`output_col` (`str`, default=`"back_transformed_cost"`)  
Name of the output column to store the final back-transformed predictions.

**Returns**  
`df_corrected` (`pyspark.sql.DataFrame`)  
A DataFrame including original columns, WLS log-cost prediction, optionally predicted variance, and the final back-transformed target variable.

**Example**  
```python
from helpers import score_with_bias_correction

# Score a new dataset using pre-trained models
scored_df = score_with_bias_correction(
    df=new_data,
    feature_cols=["age", "bmi", "income"],
    wls_model_path="./models/final_wls_model",
    variance_model_path="./models/final_variance_model",
    output_col="back_transformed_predicted_cost"
)

scored_df.select("back_transformed_predicted_cost").show()
```
### compute_regression_metrics
**Description**  
Computes common regression metrics—Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R², and adjusted R²—based on prediction and label columns in a PySpark DataFrame. Efficiently computes metrics in two passes and handles null values by dropping them first.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
Input DataFrame containing both label and prediction columns.

`label_col` (`str`)  
Name of the column containing the true label values.

`prediction_col` (`str`)  
Name of the column containing predicted values.

`n_features` (`int`)  
Number of features used in the model (for computing adjusted R²).

**Returns**  
`metrics_dict` (`dict`)  
Dictionary containing:  
- `mse`: Mean Squared Error  
- `rmse`: Root Mean Squared Error  
- `mae`: Mean Absolute Error  
- `r2`: Coefficient of Determination (R²)  
- `adjusted_r2`: Adjusted R² (or `None` if not enough data points)

**Example**  
```python
from helpers import compute_regression_metrics

# Evaluate model performance
metrics = compute_regression_metrics(
    df=scored_df,
    label_col="lgcost",
    prediction_col="prediction",
    n_features=25
)

print(metrics)
```
### plot_residual_diagnostics
**Description**  
Generates two diagnostic plots for a regression model’s residuals:  
1. A scatter plot of residuals vs. predicted values  
2. A histogram showing the distribution of residuals  
These plots help detect heteroscedasticity, skewness, or other model misspecifications. The function converts a PySpark DataFrame to Pandas for visualization.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The input DataFrame containing the prediction and label columns.

`label_col` (`str`, default=`"lgcost"`)  
The name of the actual/observed target variable column.

`prediction` (`str`, default=`"prediction"`)  
The name of the prediction column generated by a regression model.

**Returns**  
None  
Displays two plots using `matplotlib`: residuals vs. predictions and residuals histogram.

**Example**  
```python
from helpers import plot_residual_diagnostics

# Plot diagnostics for a model output DataFrame
plot_residual_diagnostics(df, label_col="actual_cost", prediction="predicted_cost")
```
### plot_cv_metric_by_experiment
**Description**  
Generates a line plot showing the performance metric (e.g., RMSE) across different cross-validation experiments. Helps visualize variability in model performance over parameter combinations or iterations. Automatically converts PySpark DataFrame to Pandas for plotting.

**Parameters**  
`df` (`pyspark.sql.DataFrame`)  
The DataFrame containing CV results. Must include columns: `experiment_num` and `avg_rmse`.

`title` (`str`, default=`"Cross-Validation RMSE by Experiment"`)  
Title to display on the plot.

**Returns**  
None  
Displays a Matplotlib line plot of experiment number vs. average RMSE.

**Example**  
```python
from helpers import plot_cv_metric_by_experiment

# Visualize CV results
plot_cv_metric_by_experiment(cv_results, title="WLS Cross-Validation RMSE")
```



## 📦 DataFrame Operations

### join_dataframes  
**Description**  
Performs an optimized join between two PySpark DataFrames on one or more key columns. Standardizes column names to lowercase and strips whitespace before joining. Removes duplicate key columns from the right-hand DataFrame (`df2`) to avoid name collisions.

**Parameters**  
`df1` (`pyspark.sql.DataFrame`)  
The first input DataFrame.

`df2` (`pyspark.sql.DataFrame`)  
The second input DataFrame.

`keys` (`list`)  
A list of column names to use as join keys. These keys must exist in both DataFrames.

`how` (`str`, default=`"inner"`)  
Type of join to perform. Supported values: `"inner"`, `"left"`, `"right"`, `"full"`, `"outer"`, `"left_outer"`, `"right_outer"`, `"full_outer"`.

**Returns**  
`joined_df` (`pyspark.sql.DataFrame`)  
The resulting DataFrame with duplicate key columns removed and rows joined according to the specified method.

**Edge Cases & Behavior**  
- Column names in both DataFrames and keys are lowercased and stripped of leading/trailing spaces.
- Raises `TypeError` if inputs are not PySpark DataFrames.
- Raises `ValueError` if keys are missing in either DataFrame or if an unsupported join type is specified.
- Automatically drops shared columns (except join keys) from the second DataFrame.
- Join keys in `df2` are temporarily renamed to avoid collision and dropped post-join.
- Warns if the result of the join is empty.

**Example**  
```python
from helpers import join_dataframes

# Example join on key columns
joined_df = join_dataframes(
    df1=left_df,
    df2=right_df,
    keys=["id", "region"],
    how="left"
)

joined_df.show()
```
### compare_dataframes 
**Description**  
Compares two PySpark DataFrames and returns a detailed report identifying schema differences, row count differences, and data content mismatches. Designed to help diagnose differences in data pipelines, testing outputs, or tracking transformations.

**Parameters**  
`df1` (`pyspark.sql.DataFrame`)  
The first PySpark DataFrame.

`df2` (`pyspark.sql.DataFrame`)  
The second PySpark DataFrame.

**Returns**  
`report` (`dict`)  
A dictionary with the following keys:  
- `"schema_diff"`:  
  - `"columns_only_in_df1"`: List of columns exclusive to `df1`.  
  - `"columns_only_in_df2"`: List of columns exclusive to `df2`.  
  - `"type_mismatches"`: Dict of common columns with differing data types.
- `"row_count"`:  
  - `"df1"`: Number of rows in `df1`.  
  - `"df2"`: Number of rows in `df2`.
- `"data_content_diff"`:  
  - `"diff_count_df1_vs_df2"`: Rows in `df1` but not in `df2`.  
  - `"diff_count_df2_vs_df1"`: Rows in `df2` but not in `df1`.  
  - `"total_diff_rows"`: Combined count of differing rows.

**Edge Cases & Behavior**  
- Raises `ValueError` if inputs are not valid PySpark DataFrames.  
- If DataFrames have no common columns, content comparison is skipped.  
- Handles empty DataFrames gracefully.  
- Accounts for null values in content comparison.  
- Catches exceptions when counting rows or comparing data.

**Example**  
```python
from helpers import compare_dataframes

comparison_report = compare_dataframes(df1, df2)

# Print schema differences
print(comparison_report["schema_diff"])

# Print row count summary
print(comparison_report["row_count"])

# Print data content differences
print(comparison_report["data_content_diff"])
```

## 📦 Saving & Loading Data

### save_to_excel 
**Description**  
Saves multiple data objects (Pandas DataFrames, PySpark DataFrames, lists, or dictionaries) into an Excel workbook, with each on a separate sheet. Accepts both positional (`*dfs`) and keyword (`**named_dfs`) arguments. Sheet names are inferred from variable names or explicitly provided keys. Automatically renames duplicate sheet names to avoid conflicts. Appends to existing Excel workbooks if the file already exists.

**Parameters**  
`output_path` (`str`)  
Directory where the Excel file will be saved.

`output_filename` (`str`)  
Name of the Excel file, should include `.xlsx`.

`*dfs` (`Union[pandas.DataFrame, pyspark.sql.DataFrame, list, dict]`)  
Positional arguments: unlabelled data inputs. Variable names are inferred from the caller’s frame to name sheets.

`**named_dfs` (`Union[pandas.DataFrame, pyspark.sql.DataFrame, list, dict]`)  
Keyword arguments: explicitly named inputs, where each key becomes the sheet name.

**Returns**  
None  
Saves data into one or more sheets of the specified Excel workbook.

**Edge Cases & Behavior**  
- Accepts data as:
  - **Pandas DataFrame**: used directly.
  - **PySpark DataFrame**: converted to Pandas.
  - **List**: converted to DataFrame with a single column `"Values"`.
  - **Dictionary**: converted to DataFrame with columns `"Key"` and `"Value"`.
- If two inputs resolve to the same sheet name, a number is appended (`Sheet`, `Sheet_1`, etc.).
- If saving to an existing Excel file, new sheets are appended without overwriting existing sheets.
- If no valid data is provided, the function exits silently.
- Raises `ValueError` if a provided object is not a supported type.

**Example**  
```python
from helpers import save_to_excel
import pandas as pd

df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"X": [9, 8], "Y": [7, 6]})
my_list = [{"foo": 1}, {"foo": 2}]
my_dict = {"name": "Alice", "age": 30}

save_to_excel(
    "./outputs", 
    "example.xlsx", 
    df1, 
    df2, 
    my_list, 
    profile=my_dict
)
```
### save_to_parquet  

**Description**  
Saves input data to disk in Parquet format with support for multiple input types (Pandas DataFrame, PySpark DataFrame, list, or dictionary). Automatically handles conversion, repartitioning, and compression. Ensures output directory is created, and overwrites if necessary. Adds a `row_id` column if converting from Pandas or list for row order traceability.

**Parameters**  
`path` (`str`)  
Directory path where the Parquet file should be saved.

`input_data` (`Union[pandas.DataFrame, pyspark.sql.DataFrame, list, dict]`)  
The data to save. Accepted formats:  
- **Pandas DataFrame**: Converted to Spark DataFrame with `row_id` added.  
- **List**: Converted to Pandas → Spark. List of dicts becomes tabular; simple list becomes a single column named `values`.  
- **Dictionary**: Treated as tabular key-value pairs, converted to Pandas → Spark.  
- **Spark DataFrame**: Saved directly, with a warning about row order.

`output_filename` (`str`)  
Name of the output file or folder (will be suffixed with `.parquet` if not already).

`spark_session` (`pyspark.sql.SparkSession`, optional)  
An active Spark session. If `None`, one is created.

`partitions` (`int`, optional)  
Number of partitions for the output. If not provided, defaults to Spark's parallelism.

`compression` (`str`, default=`"snappy"`)  
Compression codec used for writing Parquet (e.g., `"snappy"`, `"gzip"`).

`auto_partitions` (`bool`, default=`True`)  
If `True` and `partitions` is `None`, determines partition count based on cluster defaults.

`mode` (`str`, default=`"overwrite"`)  
File write mode. Options: `"overwrite"`, `"append"`, `"ignore"`, `"error"`.

**Returns**  
None  
Writes a Parquet dataset to the specified location and prints file size summary.

**Edge Cases & Behavior**  
- If the input is a **list**, it must not be empty; raises `ValueError` if so.  
- If the input is a **dictionary**, it must not be empty.  
- If `row_id` already exists in a DataFrame, a warning is issued and it is overwritten.  
- If the output folder already exists, a warning is logged and files are overwritten.  
- If a **non-supported type** is passed, a `TypeError` is raised.  
- Handles repartitioning to improve write efficiency based on cluster configuration.  
- Reports output size in MB or GB.

**Example**  
```python
from helpers import save_to_parquet

# Save a Pandas DataFrame
save_to_parquet(
    path="./output/",
    input_data=summary_df,
    output_filename="summary_data"
)

# Save a dictionary as tabular data
save_to_parquet(
    path="./output/",
    input_data={"feature": ["age", "bmi"], "coef": [0.2, 0.5]},
    output_filename="coefficients"
)

# Save a list of dicts
save_to_parquet(
    path="./output/",
    input_data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    output_filename="records"
)
```

### load_ordered_parquet  
**Description**  
Loads a single Parquet file into a PySpark DataFrame. Optionally restores row order using a `row_id` column (if present) and allows dropping that column after sorting. Useful for retrieving data saved with order-preserving logic such as `save_to_parquet`.

**Parameters**  
`data_path` (`str`)  
Directory where the Parquet file is stored.

`filename` (`str`)  
Name of the Parquet file to load (with or without `.parquet` extension).

`spark_session` (`pyspark.sql.SparkSession`, optional)`  
Existing Spark session to use. If not provided, a new one is created.

`sort_by_row_id` (`bool`, default=`True`)  
Whether to sort the DataFrame by `row_id` if the column exists.

`drop_row_id` (`bool`, default=`True`)  
If sorting by `row_id`, determines whether to drop the `row_id` column after ordering.

**Returns**  
`df` (`pyspark.sql.DataFrame`)  
The loaded DataFrame, optionally sorted and cleaned.

**Edge Cases & Behavior**  
- If `filename` is missing the `.parquet` extension, it is appended automatically.  
- If `sort_by_row_id` is `True` and the column does not exist, the data is returned unsorted.  
- If `drop_row_id` is `True`, the column is removed after sorting.  
- Assumes the file is a single Parquet file (not a directory of partitions).

**Example**  
```python
from helpers import load_ordered_parquet

# Load and sort a saved Parquet file
df = load_ordered_parquet(
    data_path="./output/",
    filename="results",
    sort_by_row_id=True,
    drop_row_id=True
)

df.show()
```

## 📦 System Utilities & Logging


### display_folder_info
**Description**  
Displays information about the contents of a directory, including subfolder names, file counts, and the number of rows in any Parquet files found. Designed to quickly summarize modeling output folders. Can be used to verify that model artifacts, metrics, and coefficients were saved as expected.

**Parameters**  
`spark` (`pyspark.sql.SparkSession`)  
An active SparkSession used to inspect Parquet file contents.

`base_path` (`str`)  
Path to the parent directory containing one or more subdirectories with Parquet files.

**Returns**  
`info_df` (`pyspark.sql.DataFrame`)  
A summary DataFrame containing:  
- `folder`: Name of the subdirectory  
- `file_count`: Number of Parquet files in the folder  
- `row_count`: Total number of rows across all Parquet files in the folder

**Example**  
```python
from helpers import display_folder_info

# Display summary of folder contents
info_df = display_folder_info(spark, base_path="./parquet_outputs/")

info_df.show()
```

### time_logger 
**Description**  
A lightweight timing decorator that logs the execution time of any function it wraps. Useful for profiling performance of data processing or modeling steps in Spark or Pandas workflows. Logs the function name and elapsed time in seconds.

**Parameters**  
None (used as a decorator)

**Returns**  
`wrapper` (`function`)  
The decorated function with timing logic added.

**Example**  
```python
from helpers import time_logger

@time_logger
def run_expensive_operation():
    # Simulate long computation
    import time
    time.sleep(2)
    return "Done"

result = run_expensive_operation()
```

