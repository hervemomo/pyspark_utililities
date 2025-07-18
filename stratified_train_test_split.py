
from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import monotonically_increasing_id

def stratified_train_test_split(
    df: DataFrame,
    strata_col: str,
    train_frac: float = 0.8,
    seed: int = 42,
    uid_col: str = "_strat_uid"
) -> Tuple[DataFrame, DataFrame]:
    """
    Perform a stratified train/test split on a PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    strata_col : str
        Name of the categorical column to stratify on.
    train_frac : float, default=0.8
        Fraction of each stratum to include in the training set (0 < train_frac < 1).
    seed : int, default=42
        Random seed for reproducibility.
    uid_col : str, default="_strat_uid"
        Temporary unique-ID column name (will be dropped).

    Returns
    -------
    train_df : DataFrame
        Stratified training set.
    test_df : DataFrame
        Stratified test set (the remaining rows).
    """
    # sanity check
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be between 0 and 1 (exclusive), got {train_frac}")

    # 1) add a unique ID
    df_uid = df.withColumn(uid_col, monotonically_increasing_id())

    # 2) find distinct strata
    strata_vals = [row[strata_col] 
                   for row in df_uid.select(strata_col).distinct().collect()]

    # 3) build fractions dict
    fractions = {val: train_frac for val in strata_vals}

    # 4) sample the training set
    train_df = df_uid.stat.sampleBy(strata_col, fractions, seed)

    # 5) anti-join to carve out the test set
    test_df = df_uid.join(train_df.select(uid_col), on=uid_col, how="left_anti")

    # 6) drop the helper column
    return train_df.drop(uid_col), test_df.drop(uid_col)