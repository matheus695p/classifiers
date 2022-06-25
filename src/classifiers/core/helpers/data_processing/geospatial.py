import logging
from typing import Dict, List

import geopandas as gpd
import pandas as pd

from classifiers.core.helpers.data_processing.general import (
    join_dfs,
    parallelize_dataframe,
)
from classifiers.core.helpers.data_processing.geometry import (
    create_buffer,
)
from classifiers.core.helpers.data_transformers.cleaning_utils import (
    filling_nans_by_fixed_value,
)

logger = logging.getLogger(__name__)


def compute_haversine_distance(
    input_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    max_diameter: float,
    drop_na_cols: List[str],
):
    """Computes distances between POI and users.

    Calculates distances from items in input_df to osm_df in a given distance threshold.

    The function will create POI-customer distance pairs, contained within a set maximum radius.
    Heads up: When the distance between the points exceeds this radius the join gives as result 0,
    since there is no intersection between the polygons.

    Args:
        input_df (pd.DataFrame): Input dataframe
        osm_df (pd.DataFrame): OSM points dataframe
        thresholds (List[int]): List of distance thresholds. The maximum value will be used to fetch the points in the ratio.
        diameter_of_the_city (float): Diameter of the city, to be used as a max distance.
        drop_na_cols: List[str]: cols to drop nulls on the OSM dataset

    Returns:
        data: (pd.DataFrame): output dataframe
    """
    logger.info(
        f"Dropping OSM rows that dont add information based on columns: {drop_na_cols}"
    )
    osm_df_clean = osm_df.dropna(subset=drop_na_cols)
    logger.info(
        f"Reduce data in: {(1 - (osm_df_clean.shape[0]/osm_df.shape[0]))*100} %"
    )

    # Renaming to be able to have the point information later
    logger.info("Creating distance buffer")
    input_df["geometry_point"] = input_df["geometry"]
    input_df["geometry"] = create_buffer(input_df["geometry"], distance=max_diameter)

    logger.info("Performing spatial join on datasets")
    new_df = osm_df_clean.sjoin(input_df, predicate="within").rename(
        columns={"geometry_point": "geometry_input_df", "geometry": "geometry_osm"}
    )

    # Calculating distance in meters
    logger.info("Calculating distance in meters")
    new_df = parallelize_dataframe(new_df, _calc_dist)

    data = new_df.drop(["geometry_input_df", "geometry_osm"], axis=1)
    return data


def _calc_dist(new_df: pd.DataFrame) -> pd.DataFrame:
    """Compute distances between two geometry GPD series.

    > We're converting the geometry columns of the new dataframe to the same projection, then
    calculating the distance between the two geometries

    Args:
      new_df (pd.DataFrame): the dataframe that contains the geometry of the input dataframe and the
    geometry of the OSM dataframe

    Returns:
      A dataframe with the distance between the two points.
    """
    new_df["distance"] = (
        gpd.GeoSeries(new_df["geometry_osm"])
        .to_crs("epsg:3857")
        .distance(gpd.GeoSeries(new_df["geometry_input_df"]).to_crs("epsg:3857"))
    )
    return new_df


def calculate_num_points(
    filt_df: pd.DataFrame, columns: List[str], threshold: int, groupby_col: str
):
    """
    Calculates number of interest points for each column in a given distance threshold

    Args:
        filt_df (pd.DataFrame): input Dataset
        columns (List[str]): List of columns with points
        threshold (int): Threshold distance in meters
        groupby_col (str): Column to use as reference to groupby

    Returns:
        pd.DataFrame
    """

    new_dfs = []
    filt_df["tmp"] = 1

    for column in columns:
        df_count = (
            filt_df.groupby([groupby_col, column])
            .count()
            .reset_index()
            .pivot(groupby_col, column, "tmp")
        )

        df_count.columns = [
            f"number_of_{col}_in_a_ratio_of_{threshold}_meters"
            for col in df_count.columns
        ]
        new_dfs.append(df_count.reset_index())

    return join_dfs(groupby_col, *new_dfs)


def calculate_min_mean_distance(
    df_with_distance: pd.DataFrame,
    columns: List[str],
    groupby_col: str,
    max_distance: int,
):
    """Compute Min and Mean distance to POIs.

    > For each POI, calculate the minimum and mean distance between the POI and the other POIs

    Args:
      df_with_distance (pd.DataFrame): the dataframe with the distance column
      columns (List[str]): the columns you want to calculate the distance for
      groupby_col (str): the column to group by. In this case, it's the "id" column.
      max_distance (int): the maximum distance to consider.
    """
    # distance dfs
    new_dfs = []
    filt_df = df_with_distance[(df_with_distance["distance"] != 0)]
    filt_df["tmp"] = 1
    # for each POI
    for column in columns:
        # create distance table
        df_distance = (
            filt_df.groupby([groupby_col, column])
            .agg(
                min_distance=pd.NamedAgg("distance", "min"),
                mean_distance=pd.NamedAgg("distance", "mean"),
            )
            .reset_index()
            .pivot(groupby_col, column)
        )
        # renaming
        df_distance.columns = ["_".join(cols) for cols in df_distance.columns]
        df_distance.fillna(max_distance, inplace=True)
        # save distaces
        new_dfs.append(df_distance.reset_index())

    # join all dfs
    joined_dfs = join_dfs(groupby_col, *new_dfs)
    return joined_dfs


def compute_geographic_features(
    df_with_distance: pd.DataFrame,
    users: pd.DataFrame,
    thresholds: List[float],
    max_diameter: float,
    columns: list,
    groupby_col: str,
) -> pd.DataFrame:
    """Compute distances features.

    It takes a dataframe with distances, a list of thresholds, the diameter of the city, a list of
    columns, and a groupby column, and returns a dataframe with the number of points for different
    thresholds, the minimum and mean distance, and the diameter of the city.

    Args:
      df_with_distance (pd.DataFrame): a dataframe with the distance between each point and each POI
      users (pd.DataFrame): Users df to get the POI
      thresholds (List[float]): list of distances to calculate the number of points within
      diameter_of_the_city (float): the diameter of the city in meters. This is used to fill in the NaN values in the dataframe.
      columns (list): list of columns to group by
      groupby_col (str): the column to group by, e.g. "geohash"

    Returns:
      A dataframe with the following columns:
        - 'min_distance_to_poi'
        - 'mean_distance_to_poi'
        - 'num_points_within_X.X km'
    """
    new_dfs = []
    # df with users ids
    user_id = users[[groupby_col]]
    user_id[groupby_col] = user_id[groupby_col].astype(str)

    for threshold in thresholds:
        logger.info(f"Calculating features for threshold: {threshold}")
        # Calculates number of points for different thresholds
        filt_df = df_with_distance[
            (df_with_distance["distance"] < threshold)
            & (df_with_distance["distance"] != 0)
        ]

        if len(filt_df) == 0:
            logger.warn("Dataset empty for this threshold. Skipping feature generation")
        else:
            count_df = calculate_num_points(filt_df, columns, threshold, groupby_col)
            # fill poi counts with zeros (if there are no points in the dataset)
            count_df[groupby_col] = count_df[groupby_col].astype(str)
            count_df = count_df.merge(user_id, on=[groupby_col], how="outer")
            count_df = filling_nans_by_fixed_value(count_df, value=0)
            new_dfs.append(count_df)

    # Calculates min/mean distance only once for the largest ratio in threshold
    df_dist = calculate_min_mean_distance(
        df_with_distance, columns, groupby_col, max_diameter
    )
    df_dist[groupby_col] = df_dist[groupby_col].astype(str)
    df_dist = df_dist.merge(user_id, on=[groupby_col], how="outer")
    # fill min and mean distances with the diameter of the city
    df_dist = filling_nans_by_fixed_value(df_dist, value=max_diameter)
    new_dfs.append(df_dist)

    data = join_dfs(groupby_col, *new_dfs)
    data = filling_nans_by_fixed_value(data, value=0)

    columns = list(data.columns)
    logger.info(f"Geospatial features created: {columns}")
    return data


def combine_geospatial_features(df: pd.DataFrame, feature_dict: Dict):
    """Combine tags on geospatial features.

    > Create each feature in the feature dictionary, by aggregating the columns in the dataframe using the
    specified aggregation type

    Args:
      df (pd.DataFrame): the dataframe you want to add the features to
      feature_dict (Dict): a dictionary of features to be created. The key is the name of the feature,
    and the value is a dictionary of parameters.

    Returns:
      A dataframe with the new features added.
    """
    for feature, params in feature_dict.items():
        agg_cols = params["agg_cols"]
        agg_type = params["agg_type"]
        df[feature] = df[agg_cols].agg(agg_type, axis=1)
    return df
