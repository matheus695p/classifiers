"""Feetaure selection pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from classifiers.core.helpers.modeling.embedding import (
    embedding_sparse_features,
)
from classifiers.core.helpers.reporting_html.reporting import (
    create_html_report,
)

from .model_input_nodes import (
    add_embedded_cols,
    data_formatting,
    encode_dfs,
    filter_churned_stores,
    nans_treatment,
    outlier_detection,
)


def create_pipeline() -> Pipeline:
    """Create the pipeline for model input table."""
    return pipeline(
        [
            # format cols and drop duplicated columns
            node(
                func=data_formatting,
                inputs=["raw_master_table", "params:model_input.id_col"],
                outputs=dict(
                    data="data_formatted", data_dict="dtypes_dict_before_embedding"
                ),
                name="data_formatting",
                tags="data_formatting",
            ),
            # nans treatment
            node(
                func=nans_treatment,
                inputs=[
                    "data_formatted",
                    "dtypes_dict_before_embedding",
                    "params:model_input.nans_treatment.data_formatted",
                ],
                outputs="data_without_nans",
                name="nans_treatment",
                tags="nans_treatment",
            ),
            node(
                func=filter_churned_stores,
                inputs=[
                    "data_without_nans",
                    "params:churn_col",
                    "params:apply_active_stores_filter",
                ],
                outputs=dict(
                    data_active_stores="data_active_stores",
                    droped_stores="droped_stores",
                ),
                name="filter_churned_stores",
                tags="filter_churned_stores",
            ),
            # embed columns
            node(
                func=embedding_sparse_features,
                inputs=[
                    "data_active_stores",
                    "params:model_input_embedding",
                ],
                outputs="embedding_dict_model_input",
                name="embedding_spare_features_model_input",
                tags="embedding_model_input",
            ),
            node(
                func=add_embedded_cols,
                inputs=[
                    "data_active_stores",
                    "embedding_dict_model_input",
                    "dtypes_dict_before_embedding",
                ],
                outputs=["master_table_embedding", "dtypes_dict"],
                name="embed_cols_model_input",
                tags="embedding_model_input",
            ),
            # encoded dict
            node(
                func=encode_dfs,
                inputs=[
                    "master_table_embedding",
                    "dtypes_dict",
                    "params:fs_params.customer_id_col",
                    "params:not_encode_cols",
                ],
                outputs=dict(
                    encoded_dict="encoded_dict_data",
                    mdt_encoded="master_table_encoded_with_outliers",
                ),
                name="encode_data",
                tags="encode_data",
            ),
            # outlier detection
            node(
                func=outlier_detection,
                inputs=[
                    "master_table_encoded_with_outliers",
                    "params:model_input.outliers_detection",
                ],
                outputs=[
                    "master_table_encoded",
                    "filtered_outliers",
                    "outliers_quantiles_used",
                ],
                name="outlier_detection",
                tags="outlier_detection",
            ),
        ],
        tags=["model_input"],
    )
