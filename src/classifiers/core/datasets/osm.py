"""GeoJSONDataSet loads and saves data to a local geojson file. The
underlying functionality is supported by geopandas, so it supports all
allowed geopandas (pandas) options for loading and saving geosjon files.
"""
import copy
import logging
from pathlib import PurePosixPath
from typing import Any, Dict, List, Union

import fsspec
import geopandas as gpd
import pandas as pd
import pyrosm
from kedro.io.core import (
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

logger = logging.getLogger(__name__)


class CustomOSMDataSet(AbstractVersionedDataSet):
    """Custom loader for OSM maps."""

    DEFAULT_LOAD_ARGS: Dict[str, Any] = {}
    DEFAULT_SAVE_ARGS = {"driver": "GeoJSON"}

    def __init__(
        self,
        filepath: str,
        bounding_boxs: List[list] = None,
        only_tags: bool = False,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        """Custom loader constructor."""
        _fs_args = copy.deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop("open_args_load", {})
        _fs_open_args_save = _fs_args.pop("open_args_save", {})
        _credentials = copy.deepcopy(credentials) or {}
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol = protocol
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        self._only_tags = only_tags

        self._bounding_boxs = bounding_boxs

        self._load_args = copy.deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)

        self._save_args = copy.deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

        _fs_open_args_save.setdefault("mode", "wb")
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save

    def _load(self) -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(
            load_path, **self._fs_open_args_load
        ) as fs_file:  # noqa F841
            geospatial_maps = []
            for city in self._bounding_boxs.keys():
                bounding_box = self._bounding_boxs[city]
                logger.info(f"Loading city: {city} /  bounding box: {bounding_box}")
                osm = pyrosm.OSM(load_path, bounding_box=bounding_box)
                tags = list(self._load_args.get("custom_filter").keys())
                if self._only_tags:
                    data = osm.get_data_by_custom_criteria(
                        **self._load_args, tags_as_columns=tags
                    )
                    geospatial_maps.append(data)
                else:
                    data = osm.get_data_by_custom_criteria(**self._load_args)
                    geospatial_maps.append(data)
            gpd_map = pd.concat(geospatial_maps)
            return gpd_map

    def _save(self, data: gpd.GeoDataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            data.to_file(fs_file, **self._save_args)
        self.invalidate_cache()

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False
        return self._fs.exists(load_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            load_args=self._load_args,
            save_args=self._save_args,
            version=self._version,
        )

    def _release(self) -> None:
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Invalidate underlying filesystem cache."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)
