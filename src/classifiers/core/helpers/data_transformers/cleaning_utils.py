"""These contains functions for cleaning data in pandas.
"""

import logging
from collections import defaultdict
from typing import Dict, List
import re
import numpy as np
import pandas as pd
from unidecode import unidecode

logger = logging.getLogger(__name__)


def _get_float_from_us_number(number: str) -> float:
    """Convert us number in a string format to float.

    It takes a string representing a number in US format (e.g. "1,234.56") and returns a float
    representing the same number

    Args:
      number (str): The number to convert.

    Returns:
      A list of dictionaries.
    """
    try:
        number = float(number)
    except Exception:
        number = number.replace(" ", "")
        number, decimal = number.split(".")
        number = float(number.replace(",", ""))
        decimal = float(decimal.replace(" ", "")) / 10 ** len(decimal)
        number += decimal
    return number


def _clean_abscissa_col(abscisa: str) -> float:
    """Clean abscissa column.

    It takes a string of the form "a+b" and returns a float of the form a+b/10^len(b).

    Args:
      abscisa (str): the column name of the abscissa

    Returns:
      A float with abscissa.
    """
    abscisa = abscisa.split("+")
    abscisa = [col.replace(" ", "") for col in abscisa]
    abscisa = float(abscisa[0]) + float(float(abscisa[1]) / 10 ** len(abscisa[1]))
    return abscisa


def _format_line_string(line: str, separator: str = "-") -> str:
    """Format the string line.
    It takes a string, splits it into a list of strings, replaces some characters in each string, and
    then joins the list of strings back into a single string

    Args:
      line (str): str = The line you want to format
      separator (str): The separator to use for the line. Defaults to -

    Returns:
      A list of strings.
    """
    lines = line.split(separator)
    lines = [
        _replace_elements(_unidecode_strings(line.replace(" ", ""))) for line in lines
    ]
    line = f"{separator}".join(lines)
    return line


def _convert_dd_mm_ss_to_decimal_dd(position: str, column: str) -> float:
    """Convert latitude and longitude dd-mm-ss to dd.dddd format.

    It takes a string in the format of "dd-mm-ss" and returns a float in the format of "dd.dddddd"

    Args:
      position (str): the string that contains the coordinates
      column (str): the column name of the dataframe

    Returns:
      A list of dictionaries.
    """
    direction = ["N" if column == "latitud" else "W"][0]
    degrees, minutes, seconds = position.split("-")
    lat = f"""{degrees}°{minutes}"{seconds}"{direction}"""
    deg, minutes, seconds, direction = re.split("[°'\"]", lat)
    position = (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60)) * (
        -1 if direction in ["W", "S"] else 1
    )
    return position


def _deduplicate_pandas_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    It transposes the dataframe, drops the duplicates, and then transposes it back

    Args:
      df (pd.DataFrame): The dataframe you want to deduplicate

    Returns:
      A dataframe with the duplicated columns removed.
    """
    columns = list(df.columns)
    initial_shape = df.shape
    initial_dtypes = (
        pd.DataFrame(df.dtypes)
        .reset_index()
        .drop_duplicates("index")
        .set_index("index")
        .to_dict()[0]
    )
    new_df = df.T.groupby(level=0).first().T
    new_df = new_df.astype(dtype=initial_dtypes)

    new_cols = list(df.columns)

    diff = list(set(columns).difference(set(new_cols)))
    final_shape = new_df.shape
    if len(diff) > 0:
        logger.info(
            f"Droping the one of following list of features because they are duplicated: {diff}"
        )
        logger.info(f"Shapes changes: {initial_shape} to {final_shape}")
    return new_df


def deduplicate_pandas(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Drop duplicates for pandas dataframe.

    Args:
       df: input data
       **kwargs: keywords feeding into the pandas `drop_duplicates`

    Returns:
       sub
    """
    logger.info(f"Dataframe shape before dedup: {df.shape}")

    sub = df.drop_duplicates(**kwargs)
    sub.reset_index(inplace=True, drop=True)

    logger.info(f"Dataframe shape after dedup: {sub.shape}")
    return sub


def enforce_custom_schema(data: pd.DataFrame, data_types: Dict) -> pd.DataFrame:
    """Apply schema to certain columns for pandas dataframe.

    Args:
       data: input data
       data_types: column names and their corresponding data types

    Returns:
       df
    """
    df = data.copy()
    schema_grp = defaultdict(list)
    for key, value in sorted(data_types.items()):
        schema_grp[value].append(key)

    # apply corresponding types
    for type_choice in schema_grp.keys():
        _apply_type(df, schema_grp, type_choice)

    return df


def _standarize_string_formats_dataframes(
    df: pd.DataFrame, columns: List
) -> pd.DataFrame:
    """standarize string formats.

    For each string a in df column. Standarize string formats.

    example:
        input: tienda del barrío de ñandú
        output: tienda_del_barrio_de_nandu

    Args:
      df (pd.DataFrame): the dataframe you want to unidecode
      columns (List): List of columns to standarize

    Returns:
      A dataframe with the columns specified in the columns parameter unidecoded.
    """
    if len(columns) > 0:
        for col in columns:
            df[col] = df[col].apply(lambda col: _unidecode_strings(col))
            df[col] = df[col].apply(lambda col: _replace_elements(col))
    return df


def _apply_type(df: pd.DataFrame, schema_grp: Dict, type_choice: str) -> None:
    """Apply schema to certain column.

    Args:
       df: input data
       schema_grp: data types and their corresponding columns
       type_choice: dtypes available
    """
    current_type_cols = schema_grp.get(type_choice)

    conversions = {
        "numeric": {"func": pd.to_numeric},
        "categorical": {"func": lambda x: x.astype("str")},
        "boolean": {"func": series_convert_bool},
        "datetime": {"func": pd.to_datetime},
    }

    df[current_type_cols] = df[current_type_cols].apply(**conversions[type_choice])


def series_convert_bool(col: pd.Series) -> pd.Series:
    """Convert series to boolean type.

    Args:
        col: column to convert

    Returns:
        pd.Series: converted col
    """
    return col.apply(convert_bool).astype("bool")


# pylint: disable=C0103
def convert_bool(value: str) -> int:
    """Convert different boolean candidates to 0 or 1.

    Args:
       value: input value

    Returns:
       0 or 1

    Raises:
        ValueError: This is raised when the value is
            invalid
    """
    v = str(value).lower()
    if v in ("yes", "true", "t", "1", "on", "1.0"):
        return 1
    if v in ("no", "false", "f", "0", "off", "0.0"):
        return 0
    if v == "nan":
        return np.NaN

    raise ValueError("Invalid boolean value %r" % (v,))


def apply_outlier_remove_rule(
    df: pd.DataFrame, rule: str, num_tag_range: Dict
) -> pd.DataFrame:
    """Remove outliers with selected rule.

    Args:
       df: input data
       rule: `clip` or `drop`
       num_tag_range: dict with col name and its value range

    Returns:
       df

    Raises:
        ValueError: This is raised when there is an invalid outlier removal rule
    """
    for col in df.columns:
        # skip columns that are not numeric
        if col not in num_tag_range.keys():
            continue

        td_low, td_up = num_tag_range[col]

        # skip columns that don't have lower and upper limits
        if np.isnan(td_low) and np.isnan(td_up):
            continue

        lower = None if np.isnan(td_low) else td_low
        upper = None if np.isnan(td_up) else td_up

        if rule == "clip":
            df[col].clip(lower, upper, inplace=True)
        elif rule == "drop":
            df[col].mask((df[col] < lower) | (df[col] > upper), inplace=True)
        else:
            raise ValueError(
                "Invalid outlier removal rule %r. "
                "Choose supported rules 'clip' or 'drop'" % (rule,)
            )

    return df


def remove_cols(df: pd.DataFrame, remove_cols: List[str]) -> pd.DataFrame:
    """
    Remove certain columns from dataset

    Args:
        df (pd.DataFrame): Input dataframe
        remove_cols (List[str]): List of columns to remove

    Returns:
        pd.DataFrame
    """

    select_cols = [col for col in df.columns if col not in remove_cols]

    return df[select_cols]


def filling_nans_by_fixed_value(df: pd.DataFrame, value: float = 0) -> pd.DataFrame:
    """fills all NaN values.

    This function fills all NaN values in a dataframe with a fixed value

    Args:
      df (pd.DataFrame): pd.DataFrame - the dataframe to be transformed
      value (float): The value to use to fill the NaN values. Defaults to 0

    Returns:
      A dataframe with all NaN values replaced by the value specified in the function.
    """
    df.replace([None, "nan", "NaN", -np.inf, np.inf], np.nan, inplace=True)
    df.fillna(value=value, inplace=True)
    return df


def _convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numerical columns to float.

    It loops through each column in the dataframe, and tries to convert the column to an integer. If it
    fails, it leaves the column as is

    Args:
      df (pd.DataFrame): the dataframe to be converted

    Returns:
      A dataframe with all the columns converted to ints.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception:
            pass
    return df


def _get_type(
    value,
    numerical_types=["float64", "float32", "float16", "int64", "int32", "int16"],
    categorical_types=["o", "object"],
):
    """Get var types.

    Args:
      value: The value to be checked.
      numerical_types: a list of all the numerical types that you want to consider as numerical.
      categorical_types: The data types that are considered categorical.

    Returns:
      The type of the data.
    """
    if value in numerical_types:
        value_type = "numeric"
    elif value in categorical_types:
        value_type = "categorical"
    else:
        value_type = "other"
    return value_type


def _replace_elements(
    somestring,
    elem_list=[
        ["á", "a"],
        ["é", "e"],
        ["í", "i"],
        ["ó", "o"],
        ["ú", "u"],
        ["ý", "y"],
        ["à", "a"],
        ["è", "e"],
        ["ì", "i"],
        ["ò", "o"],
        ["ù", "u"],
        ["ä", "a"],
        ["ë", "e"],
        ["ï", "i"],
        ["ö", "o"],
        ["ü", "u"],
        ["ÿ", "y"],
        ["â", "a"],
        ["ê", "e"],
        ["î", "i"],
        ["ô", "o"],
        ["û", "u"],
        ["ã", "a"],
        ["õ", "o"],
        ["@", "a"],
    ],
):
    """Replace elements in a string."""
    for elems in elem_list:
        somestring = str(somestring).replace(elems[0], elems[1])

    return somestring


def _df_values_type(df: pd.DataFrame, customer_id_col: str) -> dict:
    """Get dict with var types.

    It takes a dataframe as input and returns a dictionary with three keys:

    - numerical_features: a list of numerical features
    - categorical_features: a list of categorical features
    - other_type: a list of features that are neither numerical nor categorical

    The function also prints out the features in each category

    Args:
      df (pd.DataFrame): The dataframe you want to get the values type from.
      customer_id_col (str): The name of the customer id column.

    Returns:
      A dictionary with the numerical features, categorical features and other type of features.
    """
    df_types = pd.DataFrame(df.dtypes, columns=["type"])
    df_types["type_name"] = df_types["type"].apply(lambda value: _get_type(value))
    numerical_feats = list(df_types[df_types["type_name"] == "numeric"].index)
    categorical_feats = list(df_types[df_types["type_name"] == "categorical"].index)
    other_feats = list(df_types[df_types["type_name"] == "other"].index)
    # cleaned lists
    numerical_feats = [col for col in numerical_feats if col != customer_id_col]
    categorical_feats = [col for col in categorical_feats if col != customer_id_col]
    other_feats = [col for col in other_feats if col != customer_id_col]

    # outputdict
    output_dict = {
        "numerical_features": numerical_feats,
        "categorical_features": categorical_feats,
        "other_type": other_feats,
        "customer_id_col": customer_id_col,
    }
    return output_dict


def _unidecode_strings(
    somestring: str,
    characters_to_replace=[
        "(",
        ")",
        "*",
        " ",
        ":",
        ".",
        "-",
        ";",
        "<",
        "?",
        "/",
        ",",
        "'",
        "____",
        "___",
        "__",
    ],
) -> str:
    """Unidecode string.

    It takes a string, converts it to unicode, then converts it to ascii, then lowercases it, then
    replaces all the characters in the list with underscores

    Args:
      somestring (str): The string you want to unidecode.
      characters_to_replace: a list of characters to replace with an underscore

    Returns:
      A string formatted.
    """
    try:
        somestring = somestring.lower()
        u = unidecode(somestring, "utf-8")
        formated_string = unidecode(u)
        for character in characters_to_replace:
            formated_string = formated_string.replace(character, "_")
        last_underscore_index = formated_string.rindex("_")
        if last_underscore_index == len(formated_string) - 1:
            formated_string = formated_string[:-1]
        formated_string = _replace_elements(formated_string)

    except Exception:
        # logger.info("Could'nt unidecode string %s", somestring)
        formated_string = somestring
    return formated_string


def _standarize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standarize columns names.

    It takes a dataframe and returns a dataframe with the same columns, but with the column names
    standarized.

    Args:
      df (pd.DataFrame): The dataframe you want to unidecode

    Returns:
      A dataframe with the columns unidecoded.
    """
    columns = list(df.columns)
    columns = [_unidecode_strings(col) for col in columns]
    columns = [_replace_elements(col) for col in columns]
    df.columns = columns
    return df


def _drop_column_with_threshold_of_nans(
    df: pd.DataFrame, threshold: float = 99
) -> pd.DataFrame:
    """Drop columns with more than X percentage of nans.

    It takes a dataframe and a threshold, and returns a dataframe with columns that have more than the
    threshold percentage of missing values dropped

    Args:
      df (pd.DataFrame): pd.DataFrame
      threshold (float): the percentage of nans in a column that will trigger the column to be dropped.
    Defaults to 80

    Returns:
      A dataframe with the columns that have more than 80% of nans dropped.
    """
    nans = pd.DataFrame(df.isna().sum(), columns=["nans"])
    nans.sort_values(by=["nans"], ascending=False, inplace=True)
    nans["percentage"] = nans["nans"] / len(df) * 100
    nans = nans[nans["percentage"] > threshold]
    vars_to_drop = list(nans.index)
    df = df.drop(columns=vars_to_drop)
    logger.warning(
        f"Droping {len(vars_to_drop)} columns with a nans threshold greater than : {threshold} %"
    )
    logger.warning(f"Columns dropped: {vars_to_drop}")
    return df


def _drop_col_if_present(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop column if it is present.

    If the column is present in the dataframe, drop it.

    Args:
      df (pd.DataFrame): the dataframe you want to drop the column from
      col (str): The name of the column to drop.

    Returns:
      A dataframe with the column dropped if it is present.
    """
    if col in df.columns:
        df = df.drop(columns=col)
    return df


def _select_cols(df: pd.DataFrame, select_cols: List[str]):
    """Select the columns in the dataframe that are in the list of column names.

    The function takes two arguments:

    - `df`: a dataframe
    - `select_cols`: a list of column names

    The function returns a dataframe that contains only the columns in `select_cols`

    Args:
      df (pd.DataFrame): the dataframe to be processed
      select_cols (List[str]): List[str] = ['col1', 'col2', 'col3']

    Returns:
      A dataframe with only the columns specified in the list.
    """
    if len(set(select_cols)) != len(select_cols):
        logger.warning(
            "You had duplicate cols in the selection. Selecting only not duplicate cols."
        )

    return df[list(set(select_cols))]
