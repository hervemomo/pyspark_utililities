
import os
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import builtins
import psutil
import datetime
import time


import logging
from functools import wraps

from typing import Tuple, List # for hint type




def time_logger(func):
    """
    Decorator to log the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
        
        return result  # Return the function result
    
    return wrapper
