"""Load Ilicit Data.
"""
import os
import copy
import logging
from pathlib import PurePosixPath
from typing import Any, Dict, List
import fsspec
import pandas as pd
from kedro.io.core import (
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)
from classifiers.core.helpers.data_transformers.cleaning_utils import (
    _standarize_column_names,
    _standarize_string_formats_dataframes,
    _convert_dd_mm_ss_to_decimal_dd,
    _format_line_string,
    _clean_abscissa_col,
    _get_float_from_us_number,
)

logger = logging.getLogger(__name__)


class CommonCenitDataSet(AbstractVersionedDataSet):
    """Custom loader for Ilicit data."""

    DEFAULT_LOAD_ARGS: Dict[str, Any] = {"engine": "openpyxl"}
    DEFAULT_SAVE_ARGS = {}

    def __init__(
        self,
        filepath: str,
        timestamp_cols: List[str] = [],
        georeferenced_cols: List[str] = [],
        standarized_cols: List[str] = [],
        abscissa_cols: List[str] = [],
        cast_to_str_cols: List[str] = [],
        cast_to_float_cols: List[str] = [],
        pipeline_id_cols: List[str] = [],
        cast_us_number_to_float_cols: List[str] = [],
        standarize_column_names: str = False,
        compile_from_txt: bool = False,
        delete_unnamed_cols: bool = False,
        pivot_on_tag_column: Dict = None,
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
        # arguments
        self.timestamp_cols = timestamp_cols
        self.georeferenced_cols = georeferenced_cols
        self.standarized_cols = standarized_cols
        self.abscissa_cols = abscissa_cols
        self.standarize_column_names = standarize_column_names
        self.cast_to_str_cols = cast_to_str_cols
        self.cast_to_float_cols = cast_to_float_cols
        self.pipeline_id_cols = pipeline_id_cols
        self.cast_us_number_to_float_cols = cast_us_number_to_float_cols
        self.pivot_on_tag_column = pivot_on_tag_column
        self.compile_from_txt = compile_from_txt
        self.delete_unnamed_cols = delete_unnamed_cols

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )
        if self.compile_from_txt:
            self.DEFAULT_LOAD_ARGS = {}
        self._load_args = copy.deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = copy.deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        # to be defined
        _fs_open_args_save.setdefault("mode", "wb")
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save

    def _load(self) -> pd.DataFrame:
        load_path = self._get_load_path()
        logger.info(f"Load Path: {load_path}")
        if self.compile_from_txt:
            logger.info("logging from .txt")
            # compile file
            files = os.listdir(load_path)
            df = pd.DataFrame()
            for file in files:
                if ".txt" in file:
                    data = pd.read_table(str(load_path) + "/" + file)
                    df = pd.concat([df, data], axis=0)
                    df.reset_index(drop=True, inplace=True)
        else:
            df = pd.read_excel(load_path)

        # delete delete_unnamed_cols
        if self.delete_unnamed_cols:
            df.drop(
                df.columns[df.columns.str.contains("Unnamed")],
                axis=1,
                inplace=True,
            )

        # cast us numbers
        for col in self.cast_us_number_to_float_cols:
            df[col] = df[col].apply(lambda x: _get_float_from_us_number(x))

        # format pipeline ID
        for col in self.pipeline_id_cols:
            df[col] = df[col].apply(lambda x: _format_line_string(x))

        # timestamp columns
        for col in self.timestamp_cols:
            logger.info(f"Applying timestamp to {col}")
            df[col] = df[col].apply(lambda x: str(x)[0:10])
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].apply(lambda x: str(x)[0:10])
        # latitude & longitude columns
        for col in self.georeferenced_cols:
            logger.info(f"Applying georeference to {col}")
            column = col.lower()
            column = ["latitud" if "lat" in column else "longitud"][0]
            logger.info(f"Applying georeference to {col} with args {column}")
            df[col] = df[col].apply(
                lambda x: _convert_dd_mm_ss_to_decimal_dd(x, column)
            )
        # casting columns
        for col in self.abscissa_cols:
            logger.info(f"Applying eval to {col}")
            df[col] = df[col].apply(lambda x: _clean_abscissa_col(x))
        # standarize string columns
        for col in self.standarized_cols:
            logger.info(f"Applying string standarization to {col}")
            df = _standarize_string_formats_dataframes(df, self.standarized_cols)

        # for col in cast_to_float_cols
        for col in self.cast_to_float_cols:
            logger.info(f"Casting to float column: {col}")
            df[col] = df[col].apply(float)
        # for col in cast_to_str_cols
        for col in self.cast_to_str_cols:
            logger.info(f"Casting to str column: {col}")
            df[col] = df[col].apply(str)

        if self.pivot_on_tag_column is not None:
            logger.info(f"Pivoting table to create tag column")
            agg_cols = self.pivot_on_tag_column["agg_cols"]
            col_name = self.pivot_on_tag_column["col_name"]
            value_name = self.pivot_on_tag_column["value_name"]
            iterate_cols = [col for col in df.columns if col not in agg_cols]

            final_df = pd.DataFrame()
            for col in iterate_cols:
                data = df[agg_cols + [col]]
                data.rename(columns={col: value_name}, inplace=True)
                data[col_name] = col
                final_df = pd.concat([final_df, data], axis=0)
                final_df.reset_index(drop=True)
            # assign to df
            df = final_df.copy()

        # standarize column names
        if self.standarize_column_names:
            logger.info(f"Standarizing column names")
            past_cols = list(df.columns)
            df = _standarize_column_names(df)
            logger.info(f"Standarize column names")
            new_cols = list(df.columns)
            dict_ = {}
            for col, new_col in zip(past_cols, new_cols):
                dict_[col] = new_col
            logger.info(f"Renaming dictionary: {dict_}")
        return df

    def _save(self, data: pd.DataFrame) -> None:
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
