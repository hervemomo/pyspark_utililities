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


def add_interaction_columns(df, interaction_definitions):
    """
    Efficiently generates all interaction terms in a single transformation.
    """
    interaction_expressions = []
    
    for key, (cols1, cols2) in interaction_definitions.items():
        print(f'Processing interaction: {key}')
        for col1 in cols1:
            for col2 in cols2:
                if col1 in df.columns and col2 in df.columns:  # Avoid missing column errors
                    interaction_col = f"{col1}_{col2}"
                    interaction_expressions.append(F.expr(f"{col1} * {col2}").alias(interaction_col))

    df = df.select("*", *interaction_expressions)
    return df
