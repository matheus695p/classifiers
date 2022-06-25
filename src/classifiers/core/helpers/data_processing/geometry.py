from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from classifiers.core.helpers.data_transformers.cleaning_utils import (
    _drop_col_if_present,
)


def get_centroid(
    geometry: Union[BaseGeometry, gpd.GeoSeries]
) -> Union[Point, gpd.GeoSeries]:
    """Return the centroid of a geometry or a geoseries.

    Returns:
        gpd.Geoseries: Creates a column with a centroid point between latitiude and longitude.
    """
    return geometry.centroid


def create_buffer(
    geometry: Union[BaseGeometry, gpd.GeoSeries],
    distance: Union[float, np.ndarray, pd.Series],
    resolution: int = 16,
) -> Union[Polygon, gpd.GeoSeries]:
    """Create a buffer around a geometry or a series of geometries.

    Args:
      geometry (Union[BaseGeometry, gpd.GeoSeries]): The geometry to buffer.
      distance (Union[float, np.ndarray, pd.Series]): The distance to buffer the geometry by.
      resolution (int): The number of points used to approximate a circle. Defaults to 16

    Returns:
      A GeoSeries of Polygons
    """
    geometry_crs = geometry.crs
    geometry = geometry.to_crs("EPSG:3857")
    buffer = geometry.buffer(distance=distance, resolution=resolution)
    buffer = buffer.to_crs(geometry_crs)
    return buffer


def spatial_join(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    how: str = "left",
    operation: str = "intersects",
) -> gpd.GeoDataFrame:
    """Spatial join of two geodataframes.

    `spatial_join` is a function that takes two GeoDataFrames and returns a new GeoDataFrame that is the
    result of a spatial join between the two

    Args:
      gdf1 (gpd.GeoDataFrame): the first GeoDataFrame
      gdf2 (gpd.GeoDataFrame): the GeoDataFrame you want to join to
      how (str): str = "left". Defaults to left
      operation (str): The spatial operation to use. Can be one of:. Defaults to intersects

    Returns:
      A GeoDataFrame with the same number of rows as the left GeoDataFrame.
    """
    # Dropping unallowed cols
    for col in ["index_right", "index_left"]:
        gdf1 = _drop_col_if_present(gdf1, col)
        gdf2 = _drop_col_if_present(gdf2, col)
    joined_df = gpd.GeoDataFrame(gdf1).sjoin(
        gpd.GeoDataFrame(gdf2), how=how, op=operation
    )
    return joined_df


def change_crs(
    geometry: Union[gpd.GeoSeries, gpd.GeoDataFrame], epsg: int
) -> Union[gpd.GeoSeries, gpd.GeoDataFrame]:
    """Change CRS geometry to a new one EPSG code.

    `change_crs` takes a GeoSeries or GeoDataFrame and an EPSG code and returns a GeoSeries or
    GeoDataFrame with the same geometry but in the new coordinate reference system.

    Args:
      geometry (Union[gpd.GeoSeries, gpd.GeoDataFrame]): The geometry you want to change the CRS of.
      epsg (int): The EPSG code of the projection you want to convert to.

    Returns:
      A GeoDataFrame
    """
    return geometry.to_crs(epsg=epsg)


def clip_geometry(
    geometry: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    clip: Union[BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame],
) -> Union[gpd.GeoSeries, gpd.GeoDataFrame]:
    """Clip_geometry clips a geometry with another geometry.

    Args:
      geometry (Union[gpd.GeoSeries, gpd.GeoDataFrame]): Union[gpd.GeoSeries, gpd.GeoDataFrame]
      clip (Union[BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame]): Union[BaseGeometry, gpd.GeoSeries,
    gpd.GeoDataFrame]

    Returns:
      A GeoDataFrame
    """
    return geometry.clip(clip)
