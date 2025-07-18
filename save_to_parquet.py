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


from typing import Union, Optional, Mapping, List, Dict
from pyspark import SparkContext
from pyspark.sql.utils import AnalysisException

logger = logging.getLogger(__name__)

def save_to_parquet(
    path: str,
    input_data: Union[pd.DataFrame, DataFrame, List, Dict],
    output_filename: str,
    spark_session: Optional[SparkSession] = None,
    partitions: Optional[int] = None,
    compression: str = "snappy",
    auto_partitions: bool = True,
    mode: str = "overwrite"  # ── 5. EXPOSE write mode instead of hard-coding
) -> None:  # ← explicit return type
    """
    Save input data as Parquet with optional repartitioning & compression.
    """

    # 2. SPARK SESSION & PATH PREP ─
    if spark_session is None:
        spark_session = SparkSession.builder.getOrCreate()

    os.makedirs(path, exist_ok=True)  # safe directory creation

    if not output_filename.endswith(".parquet"):
        output_filename += ".parquet"

    output_dir = os.path.join(path, output_filename)
    # logger.info(f"Target Parquet directory: {output_dir}")

    if os.path.exists(output_dir):
        logger.warning(f"Overwriting existing directory: {output_dir}")

    # 3. CONVERT INPUT → SPARK DF 
    # Warn about existing row_id
    def add_row_id(df: pd.DataFrame) -> DataFrame:
        if "row_id" in df.columns:
            logger.warning("'row_id' column exists and will be overwritten")
        df = df.reset_index(drop=True)
        df["row_id"] = df.index
        return df

    if isinstance(input_data, pd.DataFrame):
        pdf = add_row_id(input_data)
        spark_df = spark_session.createDataFrame(pdf)

    elif isinstance(input_data, list):
        if not input_data:
            raise ValueError("Cannot save an empty list to Parquet.")
        if all(isinstance(item, dict) for item in input_data):
            pdf = pd.DataFrame(input_data)
        else:
            pdf = pd.DataFrame({"values": input_data})
        pdf = add_row_id(pdf)
        spark_df = spark_session.createDataFrame(pdf)

    elif isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Cannot save an empty dictionary to Parquet.")
        pdf = pd.DataFrame(input_data)
        pdf = add_row_id(pdf)
        spark_df = spark_session.createDataFrame(pdf)

    elif isinstance(input_data, DataFrame):
        logger.warning("Input is a Spark DataFrame; order not guaranteed without 'row_id'")
        spark_df = input_data

    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    # 4. PARTITION LOGIC 
    if partitions is None and auto_partitions:
        try:
            sc: SparkContext = spark_session.sparkContext
            partitions = sc.defaultParallelism  
        except Exception as e:
            logger.warning(f"Could not auto-determine partitions: {e}. Falling back to 4.")
            partitions = 4

    if partitions:
        current = spark_df.rdd.getNumPartitions()
        if partitions > current:
            spark_df = spark_df.repartition(partitions)
        else:
            spark_df = spark_df.coalesce(partitions)

    # 5. WRITE WITH ERROR HANDLING 
    try:
        spark_df.write.mode(mode).option("compression", compression).parquet(output_dir)
    except AnalysisException as ae:
        logger.error(f"Schema/write error: {ae}")
        raise
    except Exception as e:
        logger.error(f"Failed to write Parquet: {e}")
        raise

    # 6. FILE-SIZE REPORTING 
    def get_parquet_size(dir_path: str) -> int:
        total_size = 0
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    size_bytes = get_parquet_size(output_dir)
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_mb / 1024

    if size_gb >= 1:
        print(f"File saved successfully at: {output_dir} ({size_gb:.2f} GB)")
    else:
        print(f"File saved successfully at: {output_dir} ({size_mb:.2f} MB)")

