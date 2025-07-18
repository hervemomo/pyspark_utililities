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

def save_to_excel(output_path, output_filename, *dfs, **named_dfs):
    """
    Save multiple DataFrames (including Pandas, PySpark, lists, and dictionaries) to an Excel workbook with each DataFrame on a separate sheet.
    
    Parameters
    ----------
    output_path : str
        Directory path where the Excel file will be saved.
    output_filename : str
        Name of the Excel file (should include .xlsx extension).
    *dfs : DataFrame, list, dict
        DataFrames, lists, or dictionaries passed as positional arguments. The function attempts to extract the variable 
        name from the caller's local variables to use as the sheet name.
    **named_dfs : DataFrame, list, dict
        DataFrames, lists, or dictionaries passed as keyword arguments, where the key is used as the sheet name.
        e.g save_to_excel("./data", "example.xlsx", key=df)
    """
    # Ensure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Full file path
    filename = os.path.join(output_path, output_filename)
    
    # Helper function to convert lists and dictionaries to DataFrames
    def convert_to_dataframe(data):
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, DataFrame):
            return data.toPandas()
        elif isinstance(data, list):
            return pd.DataFrame(data, columns=["Values"])
        elif isinstance(data, dict):
            return pd.DataFrame(list(data.items()), columns=["Key", "Value"])
            # return pd.DataFrame(data) # if wanting tabular format with keys as columns name
        else:
            raise ValueError("Unsupported data type. Must be Pandas DataFrame, PySpark DataFrame, list, or dictionary.")
    
    # Combine both positional and keyword inputs into one dictionary.
    all_dfs = {}

    # For positional arguments, extract variable name using the caller's local frame.
    caller_frame = inspect.currentframe().f_back
    caller_locals = caller_frame.f_locals
    for i, data in enumerate(dfs):
        sheet_name = None
        df = convert_to_dataframe(data)
        for var_name, var_val in caller_locals.items():
            if var_val is data:
                sheet_name = var_name
                break
        if not sheet_name:
            sheet_name = f"Sheet_{i+1}"
        orig_name = sheet_name
        counter = 1
        while sheet_name in all_dfs:
            sheet_name = f"{orig_name}_{counter}"
            counter += 1
        all_dfs[sheet_name] = df

    # Process keyword arguments
    for key, data in named_dfs.items():
        sheet_name = key
        df = convert_to_dataframe(data)
        orig_name = sheet_name
        counter = 1
        while sheet_name in all_dfs:
            sheet_name = f"{orig_name}_{counter}"
            counter += 1
            print(f"Warning: Sheet name '{key}' is already used in the input. Using '{sheet_name}' instead.")
        all_dfs[sheet_name] = df

    if not all_dfs:
        print("No valid data provided. Exiting function.")
        return

    # Check if file exists
    file_exists = os.path.exists(filename)
    mode = 'a' if file_exists else 'w'
    writer_kwargs = {'engine': 'openpyxl', 'mode': mode}
    
    with pd.ExcelWriter(filename, **writer_kwargs) as writer:
        existing_sheets = writer.book.sheetnames if file_exists else []
        for sheet_name, df in all_dfs.items():
            orig_sheet_name = sheet_name
            new_sheet_name = sheet_name
            counter = 1
            while new_sheet_name in existing_sheets:
                new_sheet_name = f"{orig_sheet_name}_{counter}"
                counter += 1
            if new_sheet_name != orig_sheet_name:
                print(f"Warning: Sheet '{orig_sheet_name}' already exists in workbook. Saving as '{new_sheet_name}' instead.")
            df.to_excel(writer, sheet_name=new_sheet_name, index=False)
            existing_sheets.append(new_sheet_name)
    
    print(f"Data successfully saved to '{filename}'.")
