
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def one_hot_encode_spark(df, to_encode_cols) -> DataFrame:
    """
    Function to perform One-Hot Encoding on multiple categorical variables in a PySpark DataFrame.
    
    Args:
    df (pyspark.sql.DataFrame): Input DataFrame
    to_encode_cols (list): List of categorical column names to encode

    Returns:
    pyspark.sql.DataFrame: Transformed DataFrame with one-hot encoded columns
    """
 
    print("Starting One-Hot Encoding.")

    # Handle edge case: Empty column list
    if not to_encode_cols:
        print("No columns provided for encoding. Returning original DataFrame.")
        return df

    # print("\nOriginal DataFrame:")
    # df.show(5)
    # Ensure all specified columns exist in the DataFrame
    to_encode_cols = [col for col in to_encode_cols if col in df.columns]
    if not to_encode_cols:
        print("None of the specified columns exist in the DataFrame. Returning original DataFrame.")
        return df

    not_encoded_columns = [c for c in df.columns if c not in to_encode_cols]
  
    # Step 1: Convert categorical columns to indices using StringIndexer
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep", stringOrderType="alphabetAsc") for col in to_encode_cols
    ]

    # print('step 1 completed')
    
    for indexer in indexers:
        df = indexer.fit(df).transform(df)
    # print("\nIndexed DataFrame:")
    # df.show(5)

    # Step 2: Apply OneHotEncoder (dropLast=False to keep reference categories)
    encoder = OneHotEncoder(
        inputCols=[f"{col}_index" for col in to_encode_cols],
        outputCols=[f"{col}_encoded" for col in to_encode_cols],
        dropLast=False
    )
    # print('step 2 completed')
    
    df_encoded = encoder.fit(df).transform(df)
    
    # Convert sparse vectors to arrays
    for col_name in to_encode_cols:
        df_encoded = df_encoded.withColumn(f"{col_name}_array", vector_to_array(F.col(f"{col_name}_encoded")))

    # print("\nEncoded DataFrame with sparse vectors:")
    # df_encoded.show(5)
    # print('step 3 completed')    

    # Step 4: Retrieve unique category levels for each categorical column
    category_levels = {
        col: df.select(col).distinct().rdd.flatMap(lambda x: x).collect() for col in to_encode_cols
    }
    # print('step 4 completed')

    # Ensure categories are sorted (so indexes match one-hot encoded order)
    for col in category_levels:
        category_levels[col].sort()

    # Step 5: Dynamically generate encoded column names
    encoded_columns = []
    for col_name in to_encode_cols:
        encoded_columns += [
            # df_encoded[f"{col_name}_array"][i].alias(f"{col_name}_{level}")
            df_encoded[f"{col_name}_array"][i].alias(f"{level}")
            for i, level in enumerate(category_levels[col_name])
        ]
    # print("\nEncoded DataFrame with sparse arrays: ")
    # df_encoded.show(5)
    # Step 6: Select final transformed DataFrame
  
    df_final = df_encoded.select(
        *not_encoded_columns, *to_encode_cols, *encoded_columns
    )

    print("Encoding successfully completed")

    return df_final