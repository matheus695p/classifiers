import logging
import multiprocessing
from functools import reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def join_dfs(join_col, *args):
    """
    Joins multiple datasets

    Args:
        join_col (str): Column to use on join.
        *args: List of datasets to join

    Returns:
        DataFrame
    """

    dfs_list = list(args)
    first_df = dfs_list[0]

    if type(join_col) == str:
        for df in dfs_list:
            df[join_col] = df[join_col].astype(str)
    else:
        for df in dfs_list:
            for col in join_col:
                df[col] = df[col].astype(str)

    df_final = dfs_list[0]
    for df in dfs_list[1:]:
        df_final = df_final.merge(df, on=join_col, how="left")

    logger.info(f"Size increase after join: {df_final.shape[0]/first_df.shape[0]}")

    return df_final


def parallelize_dataframe(df, func, n_cores=None):
    """
    It takes a dataframe, splits it into n_cores parts, and then runs a function on each part in
    parallel

    Args:
      df: The dataframe to be parallelized.
      func: The function to apply to each dataframe chunk.
      n_cores: The number of cores you want to use.

    Returns:
      A dataframe with the same columns as the original dataframe, but with the values of the columns
        replaced by the values returned by the function.
    """
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()
    logger.info(f"Using {n_cores} to parallelize process")
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    return df


def _cast_id_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Casting id columns.

    Args:
        df (pd.DataFrame): data frame.
        col (str): col to cast as string.

    Returns:
        pd.DataFrame: cast id column.
    """
    df[col] = df[col].apply(int)
    df[col] = df[col].apply(str)
    return df


def _ilicit_valvule_position(valvule: str) -> float:
    """Cast ilicit valvule position
    It takes a string, splits it on the underscore, and returns the first element as a float

    Args:
      valvule (str): The name of the valve.

    Returns:
      The first value of the string before the underscore.
    """
    valvule = str(valvule)
    valvule = float(valvule.split("_")[0])
    return valvule


def _diameter_treatment(diameter: str) -> float:
    """Process diameter in ilicit database.

    It takes a string, tries to convert it to a float, and if it can't, it returns a NaN

    Args:
      diameter (str): The diameter of the asteroid in kilometers.

    Returns:
      A float
    """
    try:
        diameter = eval(str(diameter).replace(" ", "")[0:3])
    except Exception:
        diameter = np.nan
        logger.info("Cannot process diameter")
    return diameter
