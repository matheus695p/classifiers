"""Define the pipelines for segmentation."""

from kedro.pipeline import Pipeline, node

from classifiers.core.helpers.reporting_html.reporting import (
    create_html_report,
)


def create_pipeline() -> Pipeline:
    """Create pipeline for macro segmentation."""
    return Pipeline(
        [
            node(
                create_html_report,
                inputs=[
                    "params:feature_selection_report",
                    "demographic.feature_selection_matrix_ranked",
                    "demographic.unsupervised_feature_selection_ranking",
                    "geospatial.feature_selection_matrix_ranked",
                    "geospatial.unsupervised_feature_selection_ranking",
                    "transactional.feature_selection_matrix_ranked",
                    "transactional.unsupervised_feature_selection_ranking",
                    "product.feature_selection_matrix_ranked",
                    "product.unsupervised_feature_selection_ranking",
                    "master_table_macro_clusters",
                ],
                outputs=None,
                name="create_html_feature_selection_report",
                tags=["report", "feature_selection_report"],
            ),
        ],
        tags=["feature_selection_report"],
    )
