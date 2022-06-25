"""Model selection pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .model_selection_nodes import model_selection_ranking, optimization_results


def create_pipeline() -> Pipeline:
    """Run last model selection & feature selection optimization."""
    return pipeline(
        [
            node(
                func=model_selection_ranking,
                inputs=[
                    "master_table_encoded",
                    "unsupervised_feature_selection_ranking",
                    "params:fs_params",
                    "params:model_selection_params",
                ],
                outputs="model_selection_ranking",
                name="model_selection_ranking",
                tags="model_selection_ranking",
            ),
            node(
                func=optimization_results,
                inputs=[
                    "master_table_encoded",
                    "model_selection_ranking",
                    "params:fs_params",
                    "params:model_selection_params",
                ],
                outputs=dict(
                    model_params="model_optimized",
                    best_predictors="best_predictors",
                ),
                name="optimization_results",
                tags="optimization_results",
            ),
        ],
        tags=["model_selection"],
    )
