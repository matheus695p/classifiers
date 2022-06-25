import logging
import warnings
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from classifiers.core.helpers.objects.load import load_object

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


logger = logging.getLogger(__name__)


def embedding_sparse_features(df: pd.DataFrame, model_params: Dict) -> pd.DataFrame:
    """Create embedded features from a dimensionality reduction technique.

    It takes a dataframe and a dictionary of model parameters, and returns a dictionary of dataframes,
    each of which contains the embedded features for a particular group of columns

    Args:
      df (pd.DataFrame): The dataframe that you want to embed
      model_params (Dict): A dictionary of dictionaries. The keys are the names of the embedding models.
    The values are dictionaries with the following keys:

    Returns:
      A dictionary of dictionaries.
    """
    return_dict = {}
    for name, params in model_params.items():
        col_group = params["columns_to_embed"]
        logger.info(f"Embedding columns for: {name}. Columns: {col_group}")
        return_dict[name] = {}
        return_dict[name]["col_group"] = col_group
        data = df[col_group]
        model = load_object(params["model"])
        scaler = load_object(params["scaler"])
        pipe = Pipeline([("scaler", scaler), ("embedding", model)])
        W = pipe.fit_transform(data)
        encoded_matrix = pd.DataFrame(
            W, columns=[f"component_{i}" for i in range(W.shape[1])]
        )
        try:
            variance_explained = round(
                pipe["embedding"].explained_variance_ratio_.sum(), 3
            )
            logger.info(f"Variance Explained: {variance_explained}")
            return_dict[name]["variance_explained"] = variance_explained
        except Exception as e:
            logger.info(e)
            return_dict[name]["variance_explained"] = "Unable to calculate"

        return_dict[name]["embedded_df"] = encoded_matrix
        return_dict[name]["embedding_pipeline"] = pipe
    return return_dict
