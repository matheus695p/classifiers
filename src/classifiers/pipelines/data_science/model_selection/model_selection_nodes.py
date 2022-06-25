import logging
from pprint import pprint
from typing import Dict

import pandas as pd

from classifiers.core.helpers.modeling.explainers import (
    clustering_explainer,
)
from classifiers.core.helpers.modeling.model_optimization import (
    _clustering_optimization,
)

logger = logging.getLogger(__name__)


def model_selection_ranking(
    df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    fs_params: Dict,
    model_sel_params: Dict,
) -> pd.DataFrame:
    """Creates model selection ranking.

    Args:
        df (pd.DataFrame): Master table encoded.
        feature_importance_df (pd.DataFrame): Tree feature importance output.
        fs_params (Dict): Feature selection params.
        model_sel_params (Dict): Segmetantion params.

    Returns:
        pd.DataFrame: DataFrame with optimization ranking.
    """
    must_have_features = fs_params["must_have_features"]
    # get importance list based on mandatory features  + Tree based algorithm
    total_selected_features = must_have_features + list(
        feature_importance_df["feature_name"].values
    )
    total_selected_features = sorted(
        set(total_selected_features), key=total_selected_features.index
    )
    # clustering optimization
    model_selection_ranking = _clustering_optimization(
        df, model_sel_params, fs_params, total_selected_features
    )
    return model_selection_ranking


def optimization_results(
    df: pd.DataFrame,
    model_selection_ranking: pd.DataFrame,
    fs_params: Dict,
    model_sel_params: Dict,
) -> Dict:
    """Get best results from optimization ranking.

    Args:
        df (pd.DataFrame): master table encoded.
        model_selection_ranking (pd.DataFrame): Model selection ranking already ordered (best one should be the first).
        fs_params (Dict): Feature selection params.
        model_sel_params (Dict): Segmentation params.

    Returns:
        Dict: Model dict with best model params.
    """
    id_col = model_sel_params["id_col"]
    # get must have features
    must_have_features = fs_params["must_have_features"]
    selected_features = model_selection_ranking.iloc[[0]]["features"].iloc[0]
    # decompress list
    if type(selected_features) != list:
        selected_features = list(
            selected_features.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .split(",")
        )
        selected_features = [col.lstrip().rstrip() for col in selected_features]
    # all features
    all_feats = [id_col] + must_have_features + selected_features
    selected_features = sorted(set(all_feats), key=all_feats.index)
    # Get best model
    best_model = model_selection_ranking["model"].iloc[0]
    optimal_number_of_clusters = model_selection_ranking[
        "optimal_number_of_clusters"
    ].iloc[0]

    # scaler params
    only_features = [col for col in selected_features if col not in [id_col]]
    scaler_params = model_sel_params["clustering"]["default_scaler"]
    model_params = model_sel_params["clustering"]["model_wrapper"]["models"][best_model]
    model_params["kwargs"]["n_clusters"] = optimal_number_of_clusters
    model_params["features"] = only_features
    model_params["id_col"] = id_col
    model_params["cluster_col"] = model_sel_params["cluster_col"]
    model_params["scaler_params"] = scaler_params
    logger.info("Best model params:")
    pprint(model_params)

    best_predictors = df[only_features]
    output_dict = {"model_params": model_params, "best_predictors": best_predictors}

    return output_dict
