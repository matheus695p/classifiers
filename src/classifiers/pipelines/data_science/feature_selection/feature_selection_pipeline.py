"""Feature selection pipelines."""

from kedro.pipeline import Pipeline, node, pipeline

from .feature_selection_nodes import (
    boruta_feature_selection,
    create_final_feature_selection_ranking,
    join_rankings,
    model_based_feature_selection,
    pairwise_feature_selection,
    statistical_test_feature_selection,
    useful_information_filter,
    variance_filter_manual,
)


def create_pipeline() -> Pipeline:
    """Create all feature selection rankings + final ranking and usefull features for clustering."""
    return pipeline(
        [
            # variace filter
            node(
                func=variance_filter_manual,
                inputs=[
                    "master_table_encoded_select_features",
                    "params:fs_params",
                ],
                outputs="df_filtered_variance",
                name="variance_filter",
                tags=["variance_filter", "fs_first_filter"],
            ),
            # pairwise correlation filters
            node(
                func=pairwise_feature_selection,
                inputs=[
                    "df_filtered_variance",
                    "encoded_dict_data",
                    "params:fs_params",
                ],
                outputs=["df_pairwise_filter", "pairwise_rank"],
                name="pairwise_feature_selection",
                tags=["pairwise_feature_selection", "fs_first_filter"],
            ),
            # statistical tests ranking
            node(
                func=statistical_test_feature_selection,
                inputs=[
                    "df_pairwise_filter",
                    "params:fs_params",
                ],
                outputs="stat_test_rank",
                name="statistical_test_feature_selection",
                tags="statistical_test_feature_selection",
            ),
            # this node removes all features with non variance & and has duplicated information
            # with a selected feature
            node(
                func=useful_information_filter,
                inputs="stat_test_rank",
                outputs="fs_useful_columns",
                name="useful_information_filter",
                tags=["useful_information_filter", "fs_first_filter"],
            ),
            # boruta fs ranking
            node(
                func=boruta_feature_selection,
                inputs=[
                    "encoded_dict_data",
                    "params:fs_params",
                    "fs_useful_columns",
                ],
                outputs="boruta_rank",
                name="boruta_feature_selection",
                tags="boruta_feature_selection",
            ),
            # model based ranking.
            node(
                func=model_based_feature_selection,
                inputs=[
                    "encoded_dict_data",
                    "params:fs_params",
                    "fs_useful_columns",
                ],
                outputs="model_based_rank",
                name="model_based_feature_selection",
                tags="model_based_feature_selection",
            ),
            node(
                func=join_rankings,
                inputs=[
                    "params:fs_params",
                    *[
                        "stat_test_rank",
                        "boruta_rank",
                        "model_based_rank",
                    ],
                ],
                outputs="feature_selection_matrix",
                name="join_rankings_feature_selection",
                tags="join_rankings_feature_selection",
            ),
            # final feature selection ranking
            node(
                func=create_final_feature_selection_ranking,
                inputs=[
                    "feature_selection_matrix",
                    "encoded_dict_data",
                    "params:fs_params",
                    "fs_useful_columns",
                ],
                outputs=dict(
                    feature_selection_matrix_ranked="feature_selection_matrix_ranked",
                    target_based_fs_ranking="target_based_fs_ranking",
                ),
                name="target_based_fs_ranking",
                tags="target_based_fs_ranking",
            ),
        ],
        tags=["feature_selection"],
    )
