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


def load_ordered_parquet(
    data_path: str,
    filename: str,
    spark_session: SparkSession = None,
    sort_by_row_id: bool = True,
    drop_row_id: bool = True
) -> DataFrame:
    """
    Load a single Parquet file into a Spark DataFrame, optionally restoring order using 'row_id'.

    Args:
        data_path (str): Directory where the Parquet file is stored.
        filename (str): Name of the Parquet file (with or without '.parquet').
        spark_session (SparkSession, optional): Existing Spark session. Creates one if not provided.
        sort_by_row_id (bool): Whether to sort by 'row_id' if it exists.
        drop_row_id (bool): Whether to drop the 'row_id' column after sorting.

    Returns:
        DataFrame: The loaded Spark DataFrame.
    """
    if spark_session is None:
        spark_session = SparkSession.builder.getOrCreate()

    if not filename.endswith(".parquet"):
        filename += ".parquet"

    full_path = os.path.join(data_path.rstrip("/"), filename)

    df = spark_session.read.parquet(full_path)

    if sort_by_row_id and "row_id" in df.columns:
        df = df.orderBy("row_id")
        if drop_row_id:
            df = df.drop("row_id")

    return df
