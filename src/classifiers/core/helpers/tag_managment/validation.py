"""
Tag Dict Validation
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "tag",
    "name",
    "tag_type",
    "data_type",
    "unit",
    "range_min",
    "range_max",
    "on_off_dependencies",
    "derived",
}
UNIQUE = [{"tag", "source"}, {"name", "source"}]
COMPLETE = {"tag", "name", "data_type", "data_type_new", "source"}
KNOWN_VALUES = {
    "tag_type": {"input", "output", "state", "control", "on_off"},
    "data_type": {
        "int64",
        "object",
        "int32",
        "datetime",
        "float32",
        "float64",
        "geometry",
        "int8",
        "bool",
    },
}
# tags are checked for whether they break any of the below rules
# captured as rule - explanation
ILLEGAL_TAG_PATTERNS = [
    (r"^.*,+.*$", "no commas in tag"),
    (r"^\s.*$", "tag must not start with whitespace character"),
    (r"^.*\s$", "tag must not end with whitespace character"),
]


class TagDictError(Exception):
    """Tag Dictionary related exceptions"""


def validate_td(  # pylint:disable=too-many-locals,too-many-branches
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validates a tag dict dataframe.
    Args:
        data: tag dict data frame
    Returns:
        validated dataframe with comma separated values parsed to lists
    """
    data = data.copy()

    # check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        raise TagDictError(
            "The following columns are missing from the input dataframe: {}".format(
                missing_cols
            )
        )

    # check completeness
    for col in COMPLETE:
        if data[col].isnull().any():
            raise TagDictError("Found missing values in column `{}`".format(col))

    # check duplicates
    for cols in UNIQUE:
        logger.info(f"{cols}")
        duplicates = data.loc[data.duplicated(subset=cols), cols]
        if not duplicates.empty:
            logger.info(f"Duplicates {duplicates} from {cols}")
            raise TagDictError(
                "The following values are duplicated in column(s) `{}`: {}".format(
                    cols, list(duplicates.itertuples(index=False))
                )
            )

    # check that tag names don't contain invalid characters
    for (pattern, rule) in ILLEGAL_TAG_PATTERNS:
        matches = data.loc[data["tag"].str.match(pattern), "tag"]
        if not matches.empty:
            raise TagDictError(
                "The following tags don't adhere to rule `{}`: {}".format(
                    rule, list(matches)
                )
            )

    # valid restricted values
    for col, known_vals in KNOWN_VALUES.items():
        invalid = set(data[col].dropna()) - known_vals
        if invalid:
            raise TagDictError(
                "Found invalid entries in column {}: {}. Must be one of: {}".format(
                    col, invalid, known_vals
                )
            )

    # check on_off_dependencies
    all_tags = set(data["tag"])
    on_off_tags = set(data.loc[data["tag_type"] == "on_off", "tag"])

    on_off_dependencies = data["on_off_dependencies"]
    if not isinstance(on_off_dependencies.iloc[0], list):
        on_off_dependencies = (
            data["on_off_dependencies"]
            .fillna("")
            .apply(lambda x: [xx.strip() for xx in str(x).split(",") if xx.strip()])
        )
    for idx, deps in on_off_dependencies.items():
        not_in_tags = set(deps) - all_tags
        not_in_on_off = set(deps) - on_off_tags
        if not_in_tags:
            raise TagDictError(
                "The following on_off_dependencies of {} are not known tags: {}".format(
                    data.loc[idx, "tag"], not_in_tags
                )
            )

        if not_in_on_off:
            raise TagDictError(
                "The following on_off_dependencies of {} are not labelled as "
                "on_off type tags: {}".format(data.loc[idx, "tag"], not_in_on_off)
            )

    data["on_off_dependencies"] = on_off_dependencies

    return data
