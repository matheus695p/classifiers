"""Model input nodes.

All model input nodes, to create the reduce master table.
"""

import logging
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from classifiers.core.helpers.data_processing.general import (
    _cast_id_col,
)
from classifiers.core.helpers.data_transformers.cleaning_utils import (
    _convert_to_float,
    _deduplicate_pandas_df_columns,
    _df_values_type,
    _drop_column_with_threshold_of_nans,
    _standarize_column_names,
)
from classifiers.core.helpers.objects.load import load_object

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def data_formatting(df: pd.DataFrame, customer_id_col: str) -> Dict:
    """Data formatting processes.

    Args:
        df (pd.DataFrame): raw master table
        customer_id_col (str): the name of the column that contains the customer id.

    Returns:
        Dict: dict with data formatted to float (except categorical cols) and data types dict.
    """
    # format as float numerical cols
    df = _convert_to_float(df)
    # deduplicate columns
    df = _deduplicate_pandas_df_columns(df)
    # drop non useful columns [default nans theshold is 99%]
    df = _drop_column_with_threshold_of_nans(df)
    # dropna columns with no users
    df = df.dropna(subset=[customer_id_col])
    # validate chrun column
    df = _churn_ckeck_column(df, "churn")
    # reorder columns
    columns = [customer_id_col] + [
        col for col in list(df.columns) if col != customer_id_col
    ]
    df = df[columns]
    # data types dictionary
    data_dict = _df_values_type(df, customer_id_col)

    return dict(data=df, data_dict=data_dict)


def _churn_ckeck_column(df: pd.DataFrame, churn_col: str) -> pd.DataFrame:
    """Check churn column, for users that have not made transactions.

    Args:
      df (pd.DataFrame): the dataframe to add the churn column to
      params (dict): a dictionary of business parameters

    Returns:
      A dataframe with a new column called churn.
    """
    logger.warning("Checking churn column")
    df[churn_col] = df[churn_col].apply(float)
    # fill as churned the users that have not made transactions
    df["churn"] = df[churn_col].apply(lambda x: 1 if x > 0 else 0)
    return df


def nans_treatment(
    df: pd.DataFrame,
    dtypes_dict: Dict,
    nans_params: Dict,
):
    """Nans processing nodes.

    It takes a dataframe, a dictionary of features and their types, and nans imputer class,
    and returns a dataframe with all the missing values filled in according the methodology,
    selected on the params.

    Args:
      df (pd.DataFrame): the dataframe to be processed
      dtypes_dict (dict): a dictionary with two keys: "categorical_features"
         and "numerical_features". The values of these keys are lists of column names.
      nans_params (Dict): nans params object.

    Returns:
      A dataframe with no NaNs
    """
    # data types
    categorical_features = dtypes_dict["categorical_features"]
    numerical_features = dtypes_dict["numerical_features"]
    # replacers
    df.reset_index(drop=True, inplace=True)
    df.replace([np.inf, -np.inf, None, "nan", "NaN", "Nan"], np.nan, inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Initial Nans Sum: {df.isna().sum().sum()}")
    # numerical imputation
    numerical_imputer = load_object(nans_params["numerical"])
    numerical_imputer.fit(df[numerical_features])
    df[numerical_features] = numerical_imputer.transform(df[numerical_features])
    # categorical imputation
    cat_imputer = load_object(nans_params["categorical"])
    cat_imputer.fit(df[categorical_features])
    df[categorical_features] = cat_imputer.transform(df[categorical_features])
    logger.info(f"Final Nans Sum: {df.isna().sum().sum()}")
    return df


def add_embedded_cols(
    df: pd.DataFrame, embedding_dict: Dict, dtypes_dict: Dict
) -> pd.DataFrame:
    """
    > It takes a dataframe, a dictionary of embedding parameters, and a dictionary of dtypes, and
    returns a dataframe with the embedded columns added and updates the dtypes dictionary.

    Args:
      df (pd.DataFrame): the dataframe to add the embedded columns to
      embedding_dict (Dict): a dictionary of embedding parameters. The keys are the names of the
    embedding, and the values are dictionaries with the following keys:
      dtypes_dict (Dict): a dictionary of the data types of the features
    """
    embedded_columns = []
    cols_to_drop = []
    dfs_embedded = []
    for name, params in embedding_dict.items():
        # columns to add and drop from the selected features
        cols_to_drop.append(params["col_group"])
        df_embedded_tmp = params["embedded_df"]
        rename_cols = {col: f"{col}_{name}" for col in df_embedded_tmp.columns}
        df_embedded_tmp = df_embedded_tmp.rename(columns=rename_cols)
        dfs_embedded.append(df_embedded_tmp)
        embedded_columns.append(list(df_embedded_tmp.columns))

    embedded_columns = [item for sublist in embedded_columns for item in sublist]

    logger.info(f"Using {len(embedded_columns)} columns: {embedded_columns}")
    dtypes_dict["numerical_features"] += embedded_columns

    # add emendding the sparse features
    final_df = pd.concat([df] + dfs_embedded, axis=1)

    return final_df, dtypes_dict


def outlier_detection(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Outlier detection node.

    Apply interquantile range outlier detection technique to the numerical features.

    Args:
        df (pd.DataFrame): mdt table.
        params (Dict): outlier detection params

    Returns:
        pd.DataFrame: dataframe with outliers detected as np.nan.
    """

    outlier_cols = params["cols"]
    filt_quantile = params["filt_quantile"]
    id_col = params["id_col"]

    filtered_outliers_list = []

    quantiles_used = {}
    for col in outlier_cols:
        quantile_value = df[col].quantile(filt_quantile)
        quantiles_used[col] = quantile_value

        filt_customers = list(df[df[col] > quantile_value][id_col].unique())
        logger.info(f"Filtering outliers for col {col}, value used: {quantile_value}.")
        logger.info(f"{len(filt_customers)} outliers detected for column {col}.")

        filtered_outliers_list.append(filt_customers)

    filtered_outliers = list(
        set([item for sublist in filtered_outliers_list for item in sublist])
    )
    filt_outliers_df = df[df[id_col].isin(filtered_outliers)]
    keep_df = df[~df[id_col].isin(filtered_outliers)].reset_index(drop=True)
    logger.info(f"{len(filtered_outliers)} total outliers detected.")
    logger.info(f"Keeping {keep_df.shape[0]*100/df.shape[0]:.2f} % of the data")
    logger.info(f"Final master table shape: {keep_df.shape}")

    return keep_df, filt_outliers_df, quantiles_used


def _encode_categorical_data(
    df: pd.DataFrame, encode_categorical_cols: List, nans_threshold: float = 90
) -> pd.DataFrame:
    """Encode categorical features.

    Args:
        df (pd.DataFrame): mdt table.
        encode_categorical_cols (List): categorical columns to encode as binary.
        nans_threshold (float, optional): nans threshold. Defaults to 90.

    Returns:
        pd.DataFrame: dataframe encoded.
    """
    # encode categorical features
    if len(encode_categorical_cols) > 0:
        encoded_data = pd.get_dummies(df[encode_categorical_cols], dummy_na=False)
        encoded_data = _standarize_column_names(encoded_data)
        # drop columns with higher percentage of nans
        encoded_data = _drop_column_with_threshold_of_nans(
            encoded_data, threshold=nans_threshold
        )
        # convert to float
        encoded_data = _convert_to_float(encoded_data)
        # ensure non duplicates coluns
        encoded_data = _deduplicate_pandas_df_columns(encoded_data)
    else:
        encoded_data = pd.DataFrame()
    return encoded_data


def encode_dfs(
    df: pd.DataFrame,
    info_dict: dict,
    customer_id_col: str,
    not_encode_cols: list = [],
    nans_threshold: float = 90,
) -> dict:
    """Enconding processes.

    It takes a dataframe, a dictionary with information about data types on
    the dataframe, and the name of the customer id column. It then encodes
    the categorical features and returns a dictionary with the original
    dataframe, the dataframe with all encoded features, the dataframe
    with only numerical features, the dataframe with only categorical features,
    and the dataframe with only categorical features encoded.

    Args:
      df (pd.DataFrame): the dataframe to be encoded
      info_dict (dict): a dictionary containing the information about the dataframe.
      customer_id_col (str): the name of the column that contains the customer id
      not_encode_cols (list): List of columns not to encode
      nans_threshold (float): NaNs thresold to drop columns.

    Returns:
      A dictionary with the following keys:
        - all_encoded: dataframe with all features encoded
        - original: original dataframe
        - numerical: dataframe with only numerical features
        - categorical: dataframe with only categorical features
        - categorical_encoded: dataframe with only categorical features encoded
    """
    categorical_features = info_dict["categorical_features"]
    numerical_features = info_dict["numerical_features"]
    encode_categorical_cols = [
        f for f in categorical_features if f not in not_encode_cols
    ]
    encoded_data = _encode_categorical_data(
        df, encode_categorical_cols, nans_threshold=nans_threshold
    )
    # categorical encoded
    categorical_encoded = pd.concat([df[[customer_id_col]], encoded_data], axis=1)
    # numerical df
    df_num = df[[customer_id_col] + numerical_features]
    df_num = _standarize_column_names(df_num)
    df_num = _deduplicate_pandas_df_columns(df_num)

    # mdt with encoded features
    df_all_encoded_data = pd.concat(
        [df_num, categorical_encoded.drop(columns=customer_id_col)], axis=1
    )
    df_all_encoded_data = _standarize_column_names(df_all_encoded_data)
    df_all_encoded_data = _deduplicate_pandas_df_columns(df_all_encoded_data)
    # categorical features
    df_cat = df[[customer_id_col] + categorical_features]
    df_cat = _standarize_column_names(df_cat)
    df_cat = _deduplicate_pandas_df_columns(df_cat)

    logger.info(f"Master table encoded shape: {df_all_encoded_data.shape}")

    # cast as string
    df_all_encoded_data = _cast_id_col(df_all_encoded_data, customer_id_col)
    df = _cast_id_col(df, customer_id_col)
    df_num = _cast_id_col(df_num, customer_id_col)
    df_cat = _cast_id_col(df_cat, customer_id_col)
    categorical_encoded = _cast_id_col(categorical_encoded, customer_id_col)

    # output dict
    output_dict = {
        "mdt_encoded": df_all_encoded_data,
        "original": df,
        "numerical": df_num,
        "categorical": df_cat,
        "categorical_encoded": categorical_encoded,
    }
    return dict(encoded_dict=output_dict, mdt_encoded=df_all_encoded_data)


def filter_churned_stores(
    df: pd.DataFrame, churn_col: str, apply: str = "True"
) -> pd.DataFrame:
    """Filter churned clients from the clustring table.

    Args:
        df (pd.DataFrame): master table encoded
        churn_col (str): name of the column that contains the churn information.
        apply (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: filter data from clients.
    """
    all_clients = list(df["customer_id"].unique())
    droped_stores = []
    if bool(apply):
        df = df[df[churn_col] == 0]
        df.reset_index(drop=True, inplace=True)
        filtered_clients = list(df["customer_id"].unique())
        droped_stores = list(set(all_clients).difference(filtered_clients))

    output_dict = {"data_active_stores": df, "droped_stores": droped_stores}
    return output_dict
