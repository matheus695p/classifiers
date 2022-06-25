import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from bayes_opt import BayesianOptimization
from boruta import BorutaPy
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from classifiers.core.helpers.data_processing.general import (
    _cast_id_col,
)
from classifiers.core.helpers.data_transformers.cleaning_utils import (
    _deduplicate_pandas_df_columns,
    _drop_col_if_present,
)
from classifiers.core.helpers.modeling.explainers import (
    clustering_explainer,
)
from classifiers.core.helpers.modeling.model_optimization import (
    train_clusteiring_wrapper,
)
from classifiers.core.helpers.objects.load import load_object
from classifiers.core.helpers.reproductibility.seed_file import (
    get_global_seed,
    seed_file,
)

logger = logging.getLogger(__name__)


sns.set_style("darkgrid")
sns.set_palette("crest_r")
sns.set(rc={"figure.dpi": 100, "savefig.dpi": 100, "figure.figsize": (12, 8)})


def _point_biseroalr_corr(vector1: np.array, vector2: np.array) -> float:
    """Computes point biserial correlation.

    Args:
        a (np.array): first vector
        b (np.array): second vector

    Returns:
        float: Point biserial correlation.
    """
    return pointbiserialr(vector1, vector2)[0]


def _get_business_targets(ft_params: Dict) -> List:
    """Get business targets."""
    keys = ft_params["business_targets"].keys()
    features = []
    for key in keys:
        features += ft_params["business_targets"][key]
    features = sorted(set(features), key=features.index)
    return features


def variance_filter(
    mdt: pd.DataFrame,
    ft_params: Dict,
) -> pd.DataFrame:
    """Variance filter.

    > The function takes in a dataframe and a dictionary of parameters and returns a dataframe with the
    variance filter applied

    Args:
      mdt (pd.DataFrame): the dataframe containing the data to be applied the variance filter.
      ft_params (Dict): feature selection dict

    Returns:
      A dataframe with the columns "feature" and "variance_filter"
    """
    variance_df = pd.DataFrame(mdt.columns, columns=["feature"])
    customer_id_col = ft_params["customer_id_col"]
    threshold = ft_params["variance_filter"]["threshold"]
    business_targets = _get_business_targets(ft_params)
    data = mdt.copy()
    if customer_id_col in mdt.columns:
        mdt = mdt.drop(columns=customer_id_col)
    variance_filter = VarianceThreshold(threshold=threshold)
    variance_filter.fit_transform(mdt)
    variance_df["variance_filter"] = pd.Series(
        variance_filter.get_support(indices=False)
    )
    variance_df["variance_filter"] = variance_df["variance_filter"].apply(
        lambda x: 1 if x == True else 0
    )
    variance_df.sort_values(by=["variance_filter"], inplace=True, ascending=False)
    variance_df.reset_index(drop=True, inplace=True)

    dist = pd.DataFrame(variance_df["variance_filter"].value_counts())
    dist["percentage"] = dist["variance_filter"] / dist["variance_filter"].sum() * 100

    features = variance_df[variance_df["variance_filter"] == 1]["feature"].to_list()
    logger.info(f"Variance filter features: {features}")
    features = [customer_id_col] + features + business_targets
    features = sorted(set(features), key=features.index)
    data = data[features]
    data = _cast_id_col(data, customer_id_col)
    return data


def variance_filter_manual(
    mdt: pd.DataFrame,
    ft_params: Dict,
) -> pd.DataFrame:
    """Variance filter.

    > The function takes in a dataframe and a dictionary of parameters and returns a dataframe with the
    variance filter applied

    Args:
      mdt (pd.DataFrame): the dataframe containing the data to be applied the variance filter.
      ft_params (Dict): feature selection dict

    Returns:
      A dataframe with the columns "feature" and "variance_filter"
    """
    customer_id_col = ft_params["customer_id_col"]
    threshold = ft_params["variance_filter"]["threshold"]
    business_targets = _get_business_targets(ft_params)

    params = {"class": "sklearn.preprocessing.MinMaxScaler", "kwargs": None}
    scaler = load_object(params)
    data = mdt.copy()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    df_std = pd.DataFrame(data_scaled.std(), columns=["std"])
    df_std["variance_filter"] = df_std["std"].apply(lambda x: 1 if x > threshold else 0)
    df_std.reset_index(drop=False, inplace=True)
    df_std.rename(columns={"index": "feature"}, inplace=True)
    # dist percentage
    dist = pd.DataFrame(df_std["variance_filter"].value_counts())
    dist["percentage"] = dist["variance_filter"] / dist["variance_filter"].sum() * 100
    # features selected
    features = df_std[df_std["variance_filter"] == 1]["feature"].to_list()
    print(dist)
    logger.info(f"Variance filter features: {features}")
    features = [customer_id_col] + features + business_targets
    features = sorted(set(features), key=features.index)
    data = data[features]
    data = _cast_id_col(data, customer_id_col)
    return data


def _get_fs_ranking(
    temp_df_x_: pd.DataFrame,
    temp_df_y_: pd.DataFrame,
    df_x: pd.DataFrame,
    statistical_test: sklearn.feature_selection,
    fillna_value: float = -2,
) -> pd.DataFrame:
    """FS statistical test ranking.

    > This function takes in a dataframe of features and a dataframe of labels, and returns a dataframe
    of the features ranked by their statistical significance

    Args:
      temp_df_x_ (pd.DataFrame): The dataframe of features to be used in the statistical test.
      temp_df_y_ (pd.DataFrame): The target variable
      df_x (pd.DataFrame): The dataframe containing the features
      statistical_test (sklearn.feature_selection): sklearn.feature_selection
      fillna_value (float): float = -2,

    Returns:
      A dataframe with the rank of each feature.
    """
    if statistical_test == "f_regression":
        stat_test = f_regression
    elif statistical_test == "f_classif":
        stat_test = f_classif
    else:
        logger.error(
            f"No recognized statistical test: {stat_test}, try f_regression, f_classif, mutual_info_regression, mutual_info_classif"
        )
    # Run dependency test and get rank
    ft_selector = SelectKBest(stat_test, k="all")
    ft_selector.fit(temp_df_x_, temp_df_y_.values.ravel())
    fts_scores = ft_selector.pvalues_
    df_rank = pd.DataFrame(index=list(df_x.columns), data={"rank": fts_scores})
    df_rank = df_rank.sort_values(by="rank", ascending=False)
    df_rank = df_rank.fillna(fillna_value)
    return df_rank


def _get_vars_to_drop(upper_over_threshold: pd.DataFrame, df_rank: pd.DataFrame):
    """Select a variable to drop couples of high correlation features and lower score in the statistical test.

    > For each pair of variables in the upper triangle of the correlation matrix, if the correlation is
    1.0, then we drop the variable with the highest ranking is (lowest p-value).

    Args:
      upper_over_threshold (pd.DataFrame): a dataframe of the upper triangle of the correlation matrix
      df_rank (pd.DataFrame): a dataframe with the ranking of the variables
      verbose (bool): If True, prints out the variables that are dropped. Defaults to False

    Returns:
      A list of variables to drop.
    """
    vars_to_drop = []
    for i in range(len(upper_over_threshold.columns)):
        for j in range(len(upper_over_threshold.columns)):
            if upper_over_threshold.iloc[i, j] == 1.0:
                if (
                    df_rank.loc[upper_over_threshold.index[i], "rank"]
                    > df_rank.loc[upper_over_threshold.columns[j], "rank"]
                ):
                    vars_to_drop.append(upper_over_threshold.columns[j])
                else:
                    vars_to_drop.append(upper_over_threshold.index[i])
    vars_to_drop = list(set(vars_to_drop))
    return vars_to_drop


def _define_feature_for_pairwise_filter(
    mdt_encoded: pd.DataFrame, encoded_dict: Dict, ft_params: Dict, params: Dict
) -> List:
    """
    It takes the encoded dictionary, the feature parameters, and the parameters, and returns a list of
    the features to be used in the pairwise filter

    Args:
      mdt_encoded: pd.DataFrame
      encoded_dict (Dict): the dictionary of encoded dataframes
      ft_params (Dict): a dictionary of parameters that are used to define the features.
      params (Dict): a dictionary containing the following keys:

    Returns:
      A list of features
    """
    dataset = params["dataset"]
    customer_id_col = ft_params["customer_id_col"]
    business_targets = _get_business_targets(ft_params)
    # take fundamentals list
    fundamentals = ft_params["fundamentals"]
    # check if we have or not fundamentals
    if fundamentals == None:
        fundamentals = list(mdt_encoded.columns)

    cat_features = [
        col
        for col in list(encoded_dict["categorical_encoded"].columns)
        if col not in [customer_id_col] and col in mdt_encoded.columns
    ]
    num_features = [
        col
        for col in list(encoded_dict["numerical"].columns)
        if col not in [customer_id_col] and col in mdt_encoded.columns
    ]
    num_fundamentals = [col for col in fundamentals if col in num_features]
    cat_fundamentals = [col for col in fundamentals if col in cat_features]
    if dataset == "mdt_encoded":
        fundamentals = list(set(num_fundamentals + cat_fundamentals))
    elif dataset == "categorical_encoded":
        fundamentals = list(set(cat_fundamentals))
    elif dataset == "numerical":
        fundamentals = list(set(num_fundamentals))
    else:
        logger.error(
            "Dataset not recognized, use dataset = mdt_encoded, categorical_encoded or numerical"
        )
    fundamentals = [
        col for col in fundamentals if col not in [customer_id_col] + business_targets
    ]
    return fundamentals


def useful_information_filter(feature_rank: pd.DataFrame) -> List:
    """Get columns that have passed the dependency test theresold.

    Args:
        feature_rank (pd.DataFrame): _description_

    Returns:
        List: List of features to be used on fs.
    """
    columns = [col for col in feature_rank.columns if col not in ["feature"]]
    feature_rank["statistical_test_rank"] = feature_rank[columns].sum(axis=1)
    feature_rank.sort_values(
        by=["statistical_test_rank"], ascending=False, inplace=True
    )
    feature_rank = feature_rank[
        feature_rank["statistical_test_rank"] >= len(columns) - 1
    ]
    feature_rank.reset_index(drop=True, inplace=True)
    useful_cols = feature_rank["feature"].to_list()
    useful_cols = list(set(useful_cols))
    useful_cols = sorted(set(useful_cols), key=useful_cols.index)
    msg1 = "Columns selected after droping non variance columns, "
    msg2 = f"duplicated information and target dependency tests {useful_cols}"
    logger.info(msg1 + msg2)
    return useful_cols


def _pairwise_filter(
    mdt_encoded: pd.DataFrame,
    encoded_dict: dict,
    ft_params: dict,
    params: dict,
    key: str,
    target: str,
) -> dict:
    """Pairwise logic implementation per target.

    > The function takes the encoded dataset, the feature tools parameters, the parameters of the
    function, the key of the problem statement and the target to create a rank of the features according
    to the statistical test and the correlation matrix.

    Args:
      mdt_encoded: pd.DataFrame
      encoded_dict (dict): dict,
      ft_params (dict): the parameters of the featuretools pipeline
      params (dict): dict
      key (str): the key of the dictionary that contains the dataframe to be filtered.
      target (str): the target variable you want to predict

    Returns:
      A dictionary with the variables to drop and the useful variables.
    """
    # customer params
    statistical_test = params["statistical_test"]
    corr_method = params["corr_method"]
    corr_threshold = params["corr_threshold"]
    customer_id_col = ft_params["customer_id_col"]
    problem_statement = key.replace("to", "/").split("/")[1]
    problem_statement = [
        "classification_targets" if "cat" in problem_statement else "regression_targets"
    ][0]
    # features to use
    features_to_use = _define_feature_for_pairwise_filter(
        mdt_encoded, encoded_dict, ft_params, params
    )
    if len(features_to_use) == 0:
        return dict(vars_to_drop=[], useful_vars=[])
    # target vector
    df_y = mdt_encoded[[target]]
    # df_x only from fundamentals
    # logger.info(f"building corr matrix with: {features_to_use}")
    df_x = mdt_encoded[features_to_use]
    # drop customer id col if it is present
    df_x = _drop_col_if_present(df_x, customer_id_col)
    # drop the target from the features
    df_x = _drop_col_if_present(df_x, target)
    # statistical test ranking
    df_rank = _get_fs_ranking(df_x, df_y, df_x, statistical_test)
    # mask correlation matrix
    upper_over_threshold = _create_masked_correlation_matrix(
        df_x, df_rank, corr_method, corr_threshold
    )
    # features to drop and keep
    vars_to_drop = _get_vars_to_drop(upper_over_threshold, df_rank)
    # useful and vars to drop according to the pairwise filter.
    vars_to_drop = list(set(vars_to_drop))
    useful_vars = list(set(df_x.columns).difference(set(vars_to_drop)))
    return dict(vars_to_drop=vars_to_drop, useful_vars=useful_vars)


def _create_masked_correlation_matrix(
    df_x: pd.DataFrame, df_rank: pd.DataFrame, corr_method: str, corr_threshold: float
) -> pd.DataFrame:
    """Create masked correlation matrix.

    > It takes a dataframe of features and a dataframe of ranks, and returns a dataframe of booleans
    that indicates which features are highly correlated

    Args:
      df_x (pd.DataFrame): the dataframe containing the features
      df_rank (pd.DataFrame): The dataframe with the ranks of the features
      corr_method (str): The method used to calculate the correlation.
      corr_threshold (float): The threshold for the correlation matrix.

    Returns:
      A dataframe with the upper triangle of the correlation matrix.
    """
    # Get correlation matrix
    if corr_method == "point_biserial":
        corr_mtx = df_x.corr(method=_point_biseroalr_corr)
    else:
        corr_mtx = df_x.corr(method=corr_method)
    # correlation matrix
    corr_mtx = corr_mtx.apply(abs)
    # Get upper triagle
    upper_tri = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))
    # Drop cols over the threshold
    upper_over_threshold = upper_tri > corr_threshold
    upper_over_threshold = upper_over_threshold.where(
        np.triu(np.ones(upper_over_threshold.shape), k=1).astype(np.bool)
    )
    return upper_over_threshold


def _create_pairwise_correlation_ranking(
    feat_dict: Dict, ranking_name: str
) -> pd.DataFrame:
    """Create pairwise correlation filter.

    > It takes a dictionary of variables to drop and variables to keep, and returns a dataframe with the
    variables ranked by whether they are useful or not according to the last correlation filter.

    Args:
      feat_dict (Dict): a dictionary containing the following keys:
      ranking_name (str): The name of the ranking.

    Returns:
      A dataframe with the ranking of the features.
    """
    vars_to_drop = feat_dict["vars_to_drop"]
    useful_vars = feat_dict["useful_vars"]
    all_vars = vars_to_drop + useful_vars
    rank_i = pd.DataFrame(all_vars, columns=["feature"])
    rank_i[ranking_name] = rank_i["feature"].apply(
        lambda col: 1 if col in useful_vars else 0
    )
    rank_i.sort_values(by=[ranking_name], ascending=False, inplace=True)
    rank_i.reset_index(drop=True, inplace=True)
    return rank_i


def pairwise_feature_selection(
    mdt_encoded: pd.DataFrame, encoded_dict: Dict, ft_params: Dict
):
    """Pairwise Correlation Filter with Ranking.

    For each classification and regression target, we calculate the pairwise correlation between the
    target and all other features. We drop the one that has a lowest f_regression and f_classif score.

    Args:
      encoded_dict (Dict): the dictionary of encoded dataframes
      ft_params (Dict): This is the dictionary of parameters that we created in the previous section.

    Returns:
      A dataframe with the features and their ranking
    """
    feature_rank = pd.DataFrame(mdt_encoded.columns, columns=["feature"])

    # skip or not the feature selection step
    skip = ft_params["statistical_tests"]["skip"]
    if skip:
        logger.info(f"Skipping pairwise correlation test feature selection")
        feature_rank.fillna(value=0, inplace=True)
        columns = [col for col in feature_rank.columns if col != "feature"]
        feature_rank.sort_values(by=columns, ascending=False, inplace=True)
        feature_rank.reset_index(drop=True, inplace=True)
        data = mdt_encoded
    else:
        # for each classification and regression target
        for problem_statement in ft_params["statistical_tests"]["tests"].keys():
            # all targets for the specific problems
            if problem_statement in ft_params["business_targets"].keys():
                all_targets = ft_params["business_targets"][problem_statement]
                feature_rank = _iterate_statistical_test_rank(
                    mdt_encoded, ft_params, feature_rank, all_targets, problem_statement
                )

        feature_rank = _iterate_pairwise_correlation_ranking(
            mdt_encoded, encoded_dict, ft_params, feature_rank
        )
        cols_to_keep = _get_filtered_pairwise_columns(feature_rank, ft_params)
        customer_id = ft_params["customer_id_col"]
        business_targets = _get_business_targets(ft_params)
        features = [customer_id] + business_targets + cols_to_keep
        data = mdt_encoded[features]
    return data, feature_rank


def _get_rank_columns(feature_rank: pd.DataFrame, bt_problem: List) -> List:
    """Get columns that are in business targets and are in the rank

    Args:
        feature_rank (pd.DataFrame): feature ranking.
        bt_problem (Lit): business targets problems

    Returns:
        List: List of columns from the ranking.
    """
    columns = []
    for target in bt_problem:
        for col in feature_rank.columns:
            if target in col:
                columns.append(col)
    return columns


def _get_filtered_pairwise_columns(feature_rank: pd.DataFrame, ft_params: Dict) -> List:
    """Get columns from the pairwise correlation ranking.

    Args:
        feature_rank (pd.DataFrame): Feature ranking.
        ft_params (Dict): Feature selection params.

    Returns:
        List: List of features to use after preprocess pairwise filters.
    """
    cols_to_keep = []
    keys = list(ft_params["business_targets"].keys())
    corr_methods = _get_corr_methods(ft_params)
    for key in keys:
        bt_problem = ft_params["business_targets"][key]
        for corr_method in corr_methods:
            rank_columns = _get_rank_columns(feature_rank, bt_problem)
            rank_columns = [col for col in rank_columns if corr_method in col]
            feature_rank["pairwise_rank"] = feature_rank[rank_columns].sum(axis=1)
            threshold_pairwise = len(rank_columns)
            logger.info(f"threshold_pairwise: {threshold_pairwise}")
            feature_rank = feature_rank[
                feature_rank["pairwise_rank"] >= threshold_pairwise
            ]
            cols_to_keep += feature_rank["feature"].to_list()
    cols_to_keep = sorted(set(cols_to_keep), key=cols_to_keep.index)
    return cols_to_keep


def _get_corr_methods(ft_params: Dict) -> List:
    """Get used corr methods.

    Args:
        ft_params (Dict): feature selection params.

    Returns:
        List: correlation params
    """
    corr_params = ft_params["pairwise_correlation"]["correlation_filters"]
    corr_methods = []
    for method in corr_params.keys():
        method = corr_params[method]["corr_method"]
        if method not in corr_methods:
            corr_methods.append(method)
    return corr_methods


def _iterate_pairwise_correlation_ranking(
    mdt_encoded: pd.DataFrame,
    encoded_dict: Dict,
    ft_params: Dict,
    feature_rank: pd.DataFrame,
) -> pd.DataFrame:
    """Iterate pairwise correlation ranking over business targets.

    For each classification and regression target, we create a pairwise correlation ranking for each
    target.

    Args:
      mdt_encoded: pd.DataFrame
      encoded_dict (Dict): the dictionary of encoded features
      ft_params (Dict): This is the dictionary of parameters that you created in the previous section.
      feature_rank (pd.DataFrame): the dataframe that contains the feature ranking

    Returns:
      A dataframe with the feature ranking for each pairwise correlation filter
    """
    # for each classification and regression target
    feature_rank = pd.DataFrame(columns=["feature"])
    for key in ft_params["pairwise_correlation"]["correlation_filters"].keys():
        params = ft_params["pairwise_correlation"]["correlation_filters"][key]
        problem_statement = key.replace("to", "/").split("/")[1]
        problem_statement = [
            "classification_targets"
            if "cat" in problem_statement
            else "regression_targets"
        ][0]
        # all targets for the specific problems
        if problem_statement in ft_params["business_targets"].keys():
            all_targets = ft_params["business_targets"][problem_statement]
            for target in all_targets:
                feat_dict = _pairwise_filter(
                    mdt_encoded, encoded_dict, ft_params, params, key, target
                )
                corr_threshold = str(params["corr_threshold"]).replace(".", "")
                corr_method = params["corr_method"]
                stat_test = params["statistical_test"]
                ranking_name = f"pairwise_{corr_method}_{target}_thresh_{corr_threshold}_{stat_test}"
                msg1 = f"Creating ranking with pairwise correlation: {corr_method} / threshold: {corr_threshold} /"
                msg2 = f" rank created with statistical test: {stat_test} / business target: {target}"
                logger.info(msg1 + msg2)
                df_rank = _create_pairwise_correlation_ranking(feat_dict, ranking_name)

                # log the warning
                if df_rank.shape[0] == 0:
                    logger.info(
                        f"Empty pairwise correlation for {corr_method} with statistical test: {stat_test} / business target: {target}"
                    )
                else:
                    feature_rank = feature_rank.merge(
                        df_rank, on=["feature"], how="outer"
                    )
                    feature_rank[ranking_name].fillna(value=0, inplace=True)
                    feature_rank.sort_values(
                        by=list(feature_rank.columns), inplace=True, ascending=False
                    )
                    feature_rank.reset_index(drop=True, inplace=True)
    feature_rank.fillna(value=0, inplace=True)
    return feature_rank


def _create_boruta_rank(
    feat_selector: BorutaPy, columns: List, ranking_name: str
) -> pd.DataFrame:
    """Boruta ranking.

    > It takes a BorutaPy object, a list of column names, and a string for the ranking name, and returns
    a dataframe with the ranking of the features

    Args:
      feat_selector (BorutaPy): BorutaPy feature selection object.
      columns (List): the list of columns in the predictors dataframe.
      ranking_name (str): The name of the column that will be created in the ranking dataframe.

    Returns:
      A dataframe with the ranking of the features.
    """
    # check ranking of features
    df_rank = pd.DataFrame(columns, columns=["feature"])
    df_rank["ranking"] = pd.Series(feat_selector.support_)
    df_rank.sort_values(by=["ranking"], ascending=False, inplace=True)
    df_rank["ranking"] = df_rank["ranking"].apply(lambda x: 1 if x == True else 0)
    df_rank.reset_index(drop=True, inplace=True)
    df_rank.rename(columns={"ranking": ranking_name}, inplace=True)
    return df_rank


def _create_boruta_feat_selector(
    params: Dict, depth: List[int], problem_type: str
) -> BorutaPy:
    """Initialize borutaPy feature selection object.

    > It creates a BorutaPy object with a random forest model that has a max depth of `depth` and the
    parameters specified in `params`

    Args:
      params (Dict): Dict
      depth (List[int]): The maximum depth of the tree.
      problem_type (str): problem type, for example  "classification_targets"

    Returns:
      A BorutaPy feature selection object initialized.
    """
    if problem_type == "classification_targets":
        kwargs = params["boruta"]["random_forest_classifier"]["kwargs"]
        kwargs["max_depth"] = depth
        random_forest = RandomForestClassifier(**kwargs)
    else:
        kwargs = params["boruta"]["random_forest_regressor"]["kwargs"]
        kwargs["max_depth"] = depth
        random_forest = RandomForestRegressor(**kwargs)
    # feature selection object
    feat_selector = BorutaPy(
        random_forest, **params["boruta"]["boruta_model"]["kwargs"]
    )
    return feat_selector


def _iterate_boruta_ranking(
    mdt_encoded: pd.DataFrame,
    params: Dict,
    global_rank: pd.DataFrame,
    all_targets: List[str],
    problem_type: str,
):
    """Iterate boruta ranking over business targets.

    > For each target, create a boruta ranking for each tree depth, then merge all rankings into a
    single dataframe

    Args:
      mdt_encoded (pd.DataFrame): the dataframe with the encoded data
      params (Dict): Dict
      global_rank (pd.DataFrame): a dataframe that will be updated with the results of each boruta run
      all_targets (List[str]): list of all targets to run boruta on
      problem_type (str): classification or regression

    Returns:
      A dataframe with the ranking of the features
    """
    customer_id = params["boruta"]["customer_id_col"]
    depths = params["boruta"]["tree_depths"]
    business_targets = _get_business_targets(params)
    for target in all_targets:
        # reload fundamentals on each target
        fundamentals = params["fundamentals"]
        df_y = mdt_encoded[[target]]
        df_y = _deduplicate_pandas_df_columns(df_y)
        if fundamentals == None:
            fundamentals = [
                col
                for col in mdt_encoded.columns
                if col not in [customer_id] + business_targets
            ]
        logger.info(f"fundamentals boruta: {fundamentals}")
        df_x = mdt_encoded[fundamentals]
        df_x.reset_index(drop=True, inplace=True)
        df_x = _deduplicate_pandas_df_columns(df_x)
        columns = list(df_x.columns)
        for depth in depths:
            logger.info(
                f"Creating ranking with boruta for: {target} / RF tree depth set to {depth}"
            )
            ranking_name = f"boruta_target_{target}_depth_{depth}"
            feat_selector = _create_boruta_feat_selector(params, depth, problem_type)
            shape = df_x.shape[1]
            logger.info(f"Number of columns in boruta iteration {shape}")
            feat_selector.fit(df_x.values, df_y.values)
            df_rank = _create_boruta_rank(feat_selector, columns, ranking_name)
            global_rank = global_rank.merge(df_rank, on=["feature"], how="outer")
            global_rank[ranking_name].fillna(value=0, inplace=True)
            global_rank.sort_values(
                by=list(global_rank.columns), inplace=True, ascending=False
            )
            global_rank.reset_index(drop=True, inplace=True)
    return global_rank


def boruta_feature_selection(
    encoded_dict: Dict, fs_params: Dict, fs_useful_columns: List
) -> pd.DataFrame:
    """Perform BorutaPy feature selection over business targets

    This function takes in a dataframe of encoded features, a dictionary of parameters, and a boolean
    value for whether or not to skip the feature selection process. It then returns a dataframe of the
    features ranked by importance

    Args:
      encoded_dict (Dict[pd.DataFrame]): This is the dictionary of encoded dataframes.
      ft_params (Dict): This is a dictionary that contains all the parameters for the feature selection
    process.
      fs_useful_columns (List): Columns that have passed the variance and pairwise corrlation.

    Returns:
      A dataframe with the columns:
        - feature
        - rank_target_target_name_depth_depth
    """
    skip = fs_params["boruta"]["skip"]
    mdt_encoded = _get_mdt_useful_columns(encoded_dict, fs_params, fs_useful_columns)
    global_rank = pd.DataFrame(mdt_encoded.columns, columns=["feature"])
    customer_id = fs_params["customer_id_col"]
    if skip:
        logger.info("Skiping Boruta feature selection")
    else:
        # iteration over business targets
        for problem_type in fs_params["business_targets"].keys():
            all_targets = fs_params["business_targets"][problem_type]
            global_rank = _iterate_boruta_ranking(
                mdt_encoded, fs_params, global_rank, all_targets, problem_type
            )
    global_rank = global_rank[global_rank["feature"] != customer_id]
    return global_rank


def _apply_select_percentile(
    df_x: pd.DataFrame, df_y: pd.DataFrame, stat_test: str, percentile: float = 25
) -> List:
    """Apply SelectPercentile feature selection.

    It takes a dataframe, a statistical test, and a percentile value, and returns a list of the column
    names that are selected by the statistical test

    Args:
      df_x (pd.DataFrame): The dataframe containing the features
      df_y (pd.DataFrame): The target variable
      stat_test (str): The statistical test to use.
      percentile (float): The percentile of features to keep. Defaults to 25

    Returns:
      A list of columns that are selected by the statistical test
    """
    if stat_test == "mutual_info_classif":
        selector = SelectPercentile(mutual_info_classif, percentile=percentile)
    elif stat_test == "mutual_info_reg":
        selector = SelectPercentile(mutual_info_regression, percentile=percentile)
    else:
        list_of_tests = [
            "mutual_information_classif",
            "mutual_information_regression",
            "chi2",
        ]
        logger.error(
            f"Test not included, try one of the following list {list_of_tests}"
        )
    logger.info(f"Using statistical {stat_test} to create a percentile ranking")
    selector.fit_transform(df_x, df_y)
    cols = selector.get_support(indices=True)
    selected_columns = df_x.iloc[:, cols].columns.tolist()
    return selected_columns


def _iterate_statistical_test_rank(
    mdt_encoded: pd.DataFrame,
    ft_params: Dict,
    feature_rank: pd.DataFrame,
    all_targets: List,
    problem_statement: str,
) -> pd.DataFrame:
    """Iterate over business targets the statistical rank.

    > This function iterates through all the targets and applies the statistical tests to the dataframe

    Args:
      mdt_encoded (pd.DataFrame): the dataframe with the encoded features
      ft_params (Dict): This is the dictionary of parameters that we created in the previous section.
      feature_rank (pd.DataFrame): a dataframe with all the features and their ranks
      all_targets (List): list of all targets
      problem_statement (str): The problem statement you're trying to solve. This is used to select the
    appropriate statistical tests to run.

    Returns:
      A dataframe with the features that passed the statistical test
    """
    # Customer id col
    customer_id = ft_params["customer_id_col"]
    percentile_to_keep = ft_params["statistical_tests"]["percentile_to_keep"]
    business_targets = _get_business_targets(ft_params)
    for target in all_targets:
        # reload fundamentals on each target
        fundamentals = ft_params["fundamentals"]
        df_y = mdt_encoded[[target]]
        if fundamentals == None:
            fundamentals = [
                col
                for col in mdt_encoded.columns
                if col not in [customer_id] + business_targets
            ]
        df_x = mdt_encoded[fundamentals]
        stat_tests = ft_params["statistical_tests"]["tests"][problem_statement]
        for stat_test in stat_tests:
            ranking_name = f"stat_test_{stat_test}_tgt_{target}"
            selected_cols = _apply_select_percentile(
                df_x, df_y, stat_test, percentile=percentile_to_keep
            )
            df_selected = pd.DataFrame(selected_cols, columns=["feature"])
            df_selected[ranking_name] = 1
            feature_rank = feature_rank.merge(df_selected, on=["feature"], how="outer")
    return feature_rank


def statistical_test_feature_selection(
    mdt_encoded: pd.DataFrame,
    ft_params: Dict,
) -> pd.DataFrame:
    """Statistical test feature selection.

    For each problem statement, for each target, for each statistical test, for each feature, calculate
    statistical test rank based on mutual information scores.

    Args:
      encoded_dict (Dict[pd.DataFrame]): This is the dictionary of encoded dataframes.
      ft_params (Dict): This is a dictionary that contains all the parameters for the feature selection
    process.
      fs_useful_columns (List): Columns that have passed the variance and pairwise corrlation.

    Returns:
      A dataframe with the features and their ranks for each statistical test.
    """
    # feature rank only with useful features
    feature_rank = pd.DataFrame(mdt_encoded.columns, columns=["feature"])
    # skip or not the feature selection step
    skip = ft_params["statistical_tests"]["skip"]
    if skip:
        logger.info(f"Skipping statistical test feature selection")
    else:
        # for each classification and regression target
        for problem_statement in ft_params["statistical_tests"]["tests"].keys():
            # all targets for the specific problems
            if problem_statement in ft_params["business_targets"].keys():
                all_targets = ft_params["business_targets"][problem_statement]
                feature_rank = _iterate_statistical_test_rank(
                    mdt_encoded, ft_params, feature_rank, all_targets, problem_statement
                )
    feature_rank.fillna(value=0, inplace=True)
    columns = [col for col in feature_rank.columns if col != "feature"]
    feature_rank.sort_values(by=columns, ascending=False, inplace=True)
    feature_rank.reset_index(drop=True, inplace=True)
    return feature_rank


def _get_mdt_useful_columns(
    encoded_dict: Dict, ft_params: Dict, fs_useful_columns: List
) -> pd.DataFrame:
    """Select only the columns that passed the variance and corrlation filter.

    Select from the MDT the columns that have been selected
    in the first stage of the feature selection (pairwise + variance filters).

    Args:
        encoded_dict (Dict): master table encoded dict (mdt_encoded, numerical features and categorical features)
        ft_params (Dict): feature selection params.
        fs_useful_columns (List): Columns that have passed the variance and pairwise corrlation.

    Returns:
        pd.DataFrame: mdt with filtered columns.
    """
    customer_id_col = ft_params["customer_id_col"]
    business_targets = []
    keys = ft_params["business_targets"].keys()
    for key in keys:
        business_targets += ft_params["business_targets"][key]
    useful_cols = [customer_id_col] + fs_useful_columns + business_targets
    useful_cols = sorted(set(useful_cols), key=useful_cols.index)
    mdt_encoded = encoded_dict["mdt_encoded"]
    df = mdt_encoded[useful_cols]
    return df


def model_based_feature_selection(
    encoded_dict: Dict, ft_params: Dict, fs_useful_columns: List
) -> pd.DataFrame:
    """Model based feature selection.

    > The function takes in a dictionary of encoded dataframes and a dictionary of feature selection
    parameters. It then loops through each problem statement and target, and for each model in the model
    list, it fits a sklearn model to select features, based on feature inportance and l1 coeficientes.
    The function returns a dataframe with the features ranked by the number of times they were selected
    by the sklearn models.

    Args:
      encoded_dict (Dict[pd.DataFrame]): This is the dictionary of encoded dataframes.
      ft_params (Dict): This is a dictionary that contains all the parameters for the feature selection
    process.
      fs_useful_columns (List): Columns that have passed the variance and pairwise corrlation.

    Returns:
      A dataframe with the features and their ranking
    """
    skip = ft_params["model_based"]["skip"]
    mdt_encoded = _get_mdt_useful_columns(encoded_dict, ft_params, fs_useful_columns)
    # load mdt
    feature_rank = pd.DataFrame(mdt_encoded.columns, columns=["feature"])
    if skip:
        logger.info("Skipping model based feature selection")
    else:
        # for each classification and regression target
        for problem_statement in ft_params["business_targets"].keys():
            # all targets for the specific problems
            all_targets = ft_params["business_targets"][problem_statement]
            feature_rank = _iterate_model_based_feature_selection(
                mdt_encoded, ft_params, feature_rank, all_targets, problem_statement
            )
        cols = [col for col in feature_rank.columns if "feature" != col]
        feature_rank.sort_values(by=cols, inplace=True, ascending=False)
        feature_rank.reset_index(drop=True, inplace=True)
        feature_rank.fillna(value=0, inplace=True)
    return feature_rank


def _iterate_model_based_feature_selection(
    mdt_encoded: Dict,
    ft_params: Dict,
    feature_rank: pd.DataFrame,
    all_targets: List,
    problem_statement: str,
) -> pd.DataFrame:
    """Model based feature selection iteration.

    > Iterate through all targets and models to perform model based feature selection

    Args:
      mdt_encoded (Dict): The encoded dataframe
      ft_params (Dict): Dict = {
      feature_rank (pd.DataFrame): pd.DataFrame
      all_targets (List): List of all targets
      problem_statement (str): The problem statement you're trying to solve. This is used to select the
    models to use for feature selection.

    Returns:
      A dataframe with the feature ranking for each model
    """
    # Customer id col
    customer_id = ft_params["customer_id_col"]
    # ft_selector_kwargs
    ft_selector_kwargs = ft_params["model_based"]["feature_selector"]["kwargs"]
    # target iteration
    business_targets = _get_business_targets(ft_params)
    for target in all_targets:
        # reload fundamentals on each target
        fundamentals = ft_params["fundamentals"]
        df_y = mdt_encoded[[target]]
        if fundamentals == None:
            fundamentals = [
                col
                for col in mdt_encoded.columns
                if col not in [customer_id] + business_targets
            ]
        df_x = mdt_encoded[fundamentals]
        for model in ft_params["model_based"]["models"][problem_statement]:
            model_params = ft_params["model_based"]["models"][problem_statement][model]
            model_object = load_object(model_params)
            ranking_name = f"mb_{target}_{model}"
            logger.info(f"Model based FS {model_object} for target {target}")
            feature_selector = SelectFromModel(model_object, **ft_selector_kwargs)
            feature_selector.fit(df_x, df_y)
            df_rank = pd.DataFrame(df_x.columns, columns=["feature"])
            df_rank[ranking_name] = pd.Series(
                [1 if x == True else 0 for x in list(feature_selector.get_support())]
            )
            feature_rank = feature_rank.merge(df_rank, on=["feature"], how="outer")
    return feature_rank


def join_rankings(ft_params: Dict, *dfs: pd.DataFrame) -> pd.DataFrame:
    """Join all feature selection rankings.

    > The function takes in a dictionary of parameters and a list of dataframes. It then merges the
    dataframes on the join column specified in the parameters dictionary. It then sorts the dataframe by
    the columns and returns the dataframe

    Args:
      ft_params (Dict): a dictionary of parameters for the feature engineering process.
       (List[pd.DataFrame, pd.DataFrame]): ft_params: a dictionary of parameters for the feature

    Returns:
      A dataframe with the columns of the join_col and the columns of the dataframes that were passed
    in.
    """
    logger.info("Joining ranking dfs")
    join_col = ft_params["join_col"]
    df_ranking = dfs[0][[join_col]]
    for df in dfs:
        df_ranking = df_ranking.merge(df, on=join_col, how="outer")
    cols = [col for col in df_ranking.columns if join_col != col]
    df_ranking.sort_values(by=cols, inplace=True, ascending=False)
    df_ranking.reset_index(drop=True, inplace=True)
    df_ranking.fillna(value=0, inplace=True)
    return df_ranking


def _identify_column_type(
    feature: str, cat_fundamentals: list, num_fundamentals: list
) -> str:
    """Identify column type on the feature selection matrix.

    > If the feature is in the list of categorical fundamentals, then it's a categorical fundamental. If
    it's in the list of numerical fundamentals, then it's a numerical fundamental. Otherwise, it's a
    feature that's not used on the fundamental screen

    Args:
      feature (str): the name of the feature
      cat_fundamentals (list): list of categorical fundamental features
      num_fundamentals (list): list of numerical features

    Returns:
      A string with the feature type.
    """
    if feature in cat_fundamentals:
        feature_type = "categorical_fundamental"
    elif feature in num_fundamentals:
        feature_type = "numerical_fundamental"
    else:
        feature_type = "feature_not_used_on_fs"
    return feature_type


def _get_fundamentals_type(
    df_fs: pd.DataFrame, encoded_dict: Dict, ft_params: Dict
) -> pd.DataFrame:
    """Assign on each fundamental feature if is categorical or numeric.

    > The function takes a dataframe of features and their importance, a dictionary of encoded features,
    and a dictionary of parameters. It then returns the dataframe with a new column that identifies the
    type of feature (numerical or categorical)

    Args:
      df_fs (pd.DataFrame): the dataframe of features and scores
      encoded_dict (Dict): This is the dictionary that contains the encoded dataframes.
      ft_params (Dict): a dictionary with the following keys:
    """
    # all dataset
    mdt_encoded = encoded_dict["mdt_encoded"]
    # customer id col
    customer_id_col = ft_params["customer_id_col"]
    # take fundamentals list
    fundamentals = ft_params["fundamentals"]
    # check if we have or not fundamentals
    if fundamentals == None:
        fundamentals = list(mdt_encoded.columns)
    cat_features = [
        col
        for col in list(encoded_dict["categorical_encoded"].columns)
        if col not in [customer_id_col]
    ]
    num_features = [
        col
        for col in list(encoded_dict["numerical"].columns)
        if col not in [customer_id_col]
    ]
    # from the list of fundamentals get the numerical and categorical
    num_fundamentals = [col for col in fundamentals if col in num_features]
    cat_fundamentals = [col for col in fundamentals if col in cat_features]
    df_fs["feature_type"] = df_fs["feature"].apply(
        lambda x: _identify_column_type(x, cat_fundamentals, num_fundamentals)
    )
    return df_fs


def _extract_final_ranking(df_fs: pd.DataFrame, ft_params: Dict) -> pd.DataFrame:
    """Create final ranking following the final_ranking criteria.

    It takes a dataframe of features and their scores, and returns a dataframe of the final features
    selected.
    To change selection criteria, modify params:fs_params:final_ranking.

    Args:
      df_fs (pd.DataFrame): DataFrame with the final ranking of features
      ft_params (Dict): This is the dictionary of parameters that we created in the previous section.

    Returns:
      A dataframe with the final ranking of features
    """
    ranking_params = ft_params["final_ranking"]
    df_fs.sort_values(by=["final_rank", "feature"], ascending=False, inplace=True)
    df_fs.reset_index(drop=True, inplace=True)

    ranking_threshold = ranking_params["ranking_threshold"]
    number_of_features_threshold = ranking_params["number_of_features_threshold"]
    cond1 = (number_of_features_threshold != None) & (ranking_threshold != None)
    cond2 = (number_of_features_threshold == None) & (ranking_threshold == None)

    if cond1 or cond2:
        msg1 = "Only one of the following parameters should be None [ranking_threshold, number_of_features_threshold]"
        msg2 = "to have only one feature selection criteria"
        msg3 = ", modify the 'final_ranking' criteria on parameters:fs_params"
        logger.error(f"{msg1} {msg2}{msg3}")

    elif ranking_threshold != None:
        logger.info("Using ranking_threshold criteria for FS")
        df_rank = df_fs[df_fs["final_ranking"] >= ranking_threshold]

    elif number_of_features_threshold != None:
        logger.info("Using number_of_features_threshold criteria for FS")
        df_rank = df_fs.head(number_of_features_threshold)

    df_rank = df_rank[["feature", "feature_type", "final_rank"]]
    df_rank.sort_values(by=["final_rank", "feature"], ascending=False, inplace=True)

    final_features = df_rank["feature"].to_list()
    logger.info("===============================================================")
    logger.info(f"Final Features Selected: {final_features}")
    logger.info("===============================================================")

    return df_rank


def create_final_feature_selection_ranking(
    df_fs: pd.DataFrame,
    encoded_dict: Dict,
    ft_params: Dict,
    fs_useful_columns: List,
) -> Dict:
    """Create final feature selection ranking and select the features to be used.

    It takes the feature selection matrix and the encoded dictionary and creates a final ranking of
    features based on the business targets

    Args:
      df_fs (pd.DataFrame): This is the dataframe that contains the feature selection matrix.
      encoded_dict (Dict): This is the dictionary that contains the encoded features.
      ft_params (Dict): This is a dictionary that contains the following keys:
      fs_useful_columns (List): List with features with variance and non duplicated information.

    Returns:
      A dictionary with the following keys:
        - feature_selection_matrix_ranked: A dataframe with the feature selection matrix ranked
        - target_based_fs_ranking: A dataframe with the final ranking of the features
    """
    df_fs = _get_fundamentals_type(df_fs, encoded_dict, ft_params)
    # df_ds only features used in the feature selection process
    df_fs = df_fs[df_fs["feature_type"] != "feature_not_used_on_fs"]
    # get columns on the useful ranking.
    df_fs = df_fs[df_fs["feature"].isin(fs_useful_columns)]
    # only features with variance
    df_fs.reset_index(drop=True, inplace=True)
    fs_cols = list(df_fs.columns)
    # iteration per target
    final_rank_cols = []
    for target_type in ft_params["business_targets"].keys():
        all_targets = ft_params["business_targets"][target_type]
        for target in all_targets:
            logger.info(f"Creting final ranking for target : {target}")
            target_cols = [col for col in fs_cols if target in col]
            df_fs[f"final_{target}_ranking"] = df_fs[target_cols].sum(axis=1)
            final_rank_cols.append(f"final_{target}_ranking")
    df_fs["final_rank"] = df_fs[final_rank_cols].sum(axis=1)
    df_rank = _extract_final_ranking(df_fs, ft_params)
    df_rank.sort_values(by=["final_rank", "feature"], ascending=False, inplace=True)

    # output dict
    output_dict = {
        "feature_selection_matrix_ranked": df_fs,
        "target_based_fs_ranking": df_rank,
    }
    logger.info("Supervised ranking:")
    return output_dict


def stratified_kfold_score(
    clf: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_fold: int,
    random_state: int = 42,
) -> float:
    """Compute stratified kfold F1 score.

    Args:
        clf (sklearn.model): Classifier model.
        X (pd.DataFrame): Dataframe with predictors.
        y (pd.Series): Target.
        n_fold (int): Number of folds on cross validation.
        random_state (int, optional): Random seed of the split. Defaults to 42.

    Returns:
        float: stratified kfold F1 score
    """
    # get the metrics value
    X, y = X.values, y.values
    strat_kfold = StratifiedKFold(
        n_splits=n_fold, shuffle=False, random_state=random_state
    )
    metric_list = []
    for train_index, test_index in strat_kfold.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(x_train_fold, y_train_fold)
        preds = clf.predict(x_test_fold)
        test_metrics = f1_score(preds, y_test_fold, average="weighted")
        metric_list.append(test_metrics)
    return np.array(metric_list).mean()


def perform_bayesian_optimization(
    df: pd.DataFrame,
    fs_params: Dict,
    target_based_fs_ranking: pd.DataFrame,
    model_name: str,
) -> Dict:
    """Perform bayesian optimization on a RF classifier.

    Args:
        df (pd.DataFrame): Master table encoded.
        fs_params (Dict): Feature selection params.
        target_based_fs_ranking (pd.DataFrame): Target based end ranking.
        model_name (str): Model name to perform bayesian optimization.

    Returns:
        Dict: Output dict with bayesian optimizer + Best model params.
    """
    # seed file
    seed = get_global_seed()
    seed_file(seed, verbose=False)
    # params
    id_col = fs_params["customer_id_col"]
    target_col = fs_params["cluster_col"]
    # initial receommendation from feature selection
    clustering_columns = list(target_based_fs_ranking["feature"].values) + [id_col]
    # predictors names
    predictor_names = [
        col for col in clustering_columns if col not in [target_col, id_col]
    ]
    # number of bayesian iterations
    n_iter = fs_params["bayesian_optimization"][model_name]["number_of_iterations"]
    number_of_k_folds = fs_params["bayesian_optimization"][model_name][
        "number_of_k_folds"
    ]
    bayesian_model_params = fs_params["bayesian_optimization"][model_name]["model"]
    random_state = bayesian_model_params["kwargs"]["random_state"]
    acq = fs_params["bayesian_optimization"][model_name]["acq"]
    # list of params that should be int
    cast_int_params = fs_params["bayesian_optimization"]["cast_int_params"]
    # number of clusters
    n_clusters = df[target_col].nunique()
    f1_score_mean = round(100 / n_clusters, 2)
    msg = f"Building a classifier for {n_clusters} clusters ==> F1 score with random decisions is {f1_score_mean} [%]"
    logger.warning(msg)

    # objective function, this should be inside the node because uses gloabl vars
    if bayesian_model_params["class"] == "sklearn.ensemble.RandomForestClassifier":
        logger.info(
            f"Defining Random Forest optimization function for the model {model_name}"
        )

        def bayesian_objective_function(
            max_samples: float, n_estimators: int, max_features: float
        ):
            """Define bayesian optimization function.

            Define function with model params.

            Args:
                max_samples (float): Max samples params [0, 1]
                n_estimators (int): Number of estimators.
                max_features (float): Max samples params [0, 1]

            Returns:
                score (float): stratified kfold F1 score.
            """
            params = {
                "max_samples": max_samples,
                "max_features": max_features,
                "n_estimators": int(n_estimators),
                "class_weight": "balanced",
                "random_state": bayesian_model_params["kwargs"]["random_state"],
            }

            classifier = load_object(bayesian_model_params)
            classifier.set_params(**params)
            score = stratified_kfold_score(
                classifier,
                df[predictor_names],
                df[target_col],
                number_of_k_folds,
                random_state,
            )
            return score

    elif bayesian_model_params["class"] == "sklearn.tree.DecisionTreeClassifier":
        logger.info(
            f"Defining tree based optimization function for the model {model_name}"
        )

        def bayesian_objective_function(
            max_depth: int,
            min_samples_split: int,
            max_features: float,
            ccp_alpha: float,
        ):
            """Decision Tree params.

            Args:
                max_depth (int): Max tree depth.
                min_samples_split (int): Min number of samples to split the node.
                max_features (float): Number of features to consider each time to make the split decision.
                ccp_alpha (float): Minimal Cost-Complexity Pruning.

            Returns:
                score (float): stratified kfold F1 score.
            """
            params = {
                "max_depth": int(max_depth),
                "max_features": max_features,
                "min_samples_split": int(min_samples_split),
                "ccp_alpha": ccp_alpha,
                "class_weight": "balanced",
                "criterion": "gini",
                "random_state": bayesian_model_params["kwargs"]["random_state"],
            }
            classifier = load_object(bayesian_model_params)
            classifier.set_params(**params)
            score = stratified_kfold_score(
                classifier,
                df[predictor_names],
                df[target_col],
                number_of_k_folds,
                random_state,
            )
            return score

    else:
        logger.error(
            f"{model_name} not allowed, try feature_importance or explianer_model"
        )
    # BEGIN OPTIMIZATION
    # bayesian optimization function
    bayesian_opt = BayesianOptimization(
        bayesian_objective_function,
        fs_params["bayesian_optimization"][model_name]["exploration_space"],
        random_state=bayesian_model_params["kwargs"]["random_state"],
    )
    # run bayesian optimization
    _ = bayesian_opt.maximize(n_iter=n_iter, acq=acq)
    # extract best params
    best_model_params = bayesian_opt.max["params"]
    best_model_params = _cast_param_to_int(best_model_params, cast_int_params)
    best_model_params["random_state"] = random_state

    # model params
    model_params = {
        "class": bayesian_model_params["class"],
        "kwargs": best_model_params,
    }
    logger.info(f"Best params bayesian otpmization {model_params} for {model_name}")
    # output best model params + bayesian optimizer (to visualize bayesian exploration)
    output_dict = {
        "best_model_params": model_params,
        "bayesian_optimizer": bayesian_opt,
    }
    return output_dict


def _cast_param_to_int(params: Dict, cast_int_params: List) -> Dict:
    """Cast to int parameters that should be always ints.

    Args:
        params (Dict): Model params dictionary
        cast_int_params (List): List of params that should be int.

    Returns:
        Dict: model params casted.
    """
    for parameter in cast_int_params:
        if parameter in params.keys():
            params[parameter] = int(params[parameter])
    return params


def bayesian_optimization_summary(
    optimizer: Any, target_name: str = "f1_score"
) -> plt.Figure:
    """Summary with the bayesian optimization.

    Args:
        optimizer (Any): Bayesian optimizer object.
        target_name (str, optional): Name of the target score. Defaults to "f1_score".

    Returns:
        plt.Figure: Matplotlib figure.
    """
    opt_results = _decompress_optimizer_results(optimizer, target_name=target_name)
    fig = _plot_bayesian_opt(opt_results)
    return fig


def _decompress_optimizer_results(
    optimizer: Any, target_name: str = "f1_score"
) -> pd.DataFrame:
    """Decompress bayesian optimizer results.

    Args:
        optimizer (Any): Bayesian optimizer model
        target_name (str, optional): Name of the target score. Defaults to "f1_score".

    Returns:
        pd.DataFrame: _description_
    """
    opt_results = []
    for res in optimizer.res:
        params = res["params"]
        params["target"] = res["target"]
        opt_results.append(params)
    opt_results = pd.DataFrame(opt_results)
    opt_results.rename(columns={"target": target_name}, inplace=True)
    return opt_results


def _plot_bayesian_opt(
    opt_results: pd.DataFrame, target_name: str = "f1_score"
) -> plt.Figure:
    """Create summary plot of the bayesian optimization.

    Args:
        opt_results (pd.DataFrame): Dataframe with params and target score per iteration.
        target_name (str, optional): Name of the target score. Defaults to "f1_score".

    Returns:
        plt.Figure: Matplotlib figure with bayesian opt summary.
    """

    pd.options.plotting.backend = "matplotlib"

    parameters = [col for col in opt_results.columns if col not in [target_name]]
    fig, ax = plt.subplots(int(len(parameters) * 2), figsize=(12, 14))
    fig_counter = 0
    for x_name in parameters:
        # define vectors to plot
        opt_results.sort_values(by=x_name, inplace=True, ascending=False)
        x = opt_results[x_name].values
        y = opt_results[target_name].values
        # plot the results
        scatt = ax[fig_counter].scatter(
            x,
            y,
            lw=1,
            c=list(opt_results.index),
            cmap=plt.cm.get_cmap("inferno"),
        )

        ax[fig_counter].set_title(
            f"Bayesian Exploration Space: *{x_name}* parameter / F1_score ( {x_name} )"
        )
        ax[fig_counter].set_ylabel("F1 Score")
        ax[fig_counter].set_xlabel(f"Parameter: {x_name}")
        # add another ax
        fig_counter += 1
        opt_results.plot(ax=ax[fig_counter], x=x_name, y="f1_score", marker="o")
        ax[fig_counter].set_title(f"Bayesian F1_score ( {x_name} ) proxy")
        ax[fig_counter].set_ylabel("F1 Score")
        ax[fig_counter].set_xlabel(f"Parameter: {x_name}")
        # add another ax
        fig_counter += 1
    fig.colorbar(scatt, label="Iteration Number", orientation="horizontal")
    fig.tight_layout(pad=1.5)
    return fig


def train_clustering_wrapper_with_supervised_rank(
    df: pd.DataFrame, target_based_fs_ranking: pd.DataFrame, fs_params, model_sel_params
) -> pd.DataFrame:
    """Tree based feature selection.

    1. Take the feature selection ranking as input an train a clustering wrapper with it.

    Classifiers are:
        1. Bagging or Boosting algorithms.

    Args:
        df (pd.DataFrame): Master table encoded
        target_based_fs_ranking (pd.DataFrame): Final feature selection ranking.
        fs_params (_type_): Feature selection params
        model_sel_params (_type_): Segmentation params

    Returns:
        pd.DataFrame: Tree based feature selection
    """
    # params
    id_col = model_sel_params["id_col"]
    # initial receommendation from feature selection
    initial_recommended_features = list(target_based_fs_ranking["feature"].values) + [
        id_col
    ]
    # metrics ponderation
    metrics_computation = fs_params["clustering_iteration_trees_importance_based"][
        "metrics_computation"
    ]
    # train a clustering wrapper --> Get best model predictions
    wrapper_mdoel_dict = train_clusteiring_wrapper(
        df, model_sel_params, metrics_computation, initial_recommended_features
    )
    df_with_initial_clusters = wrapper_mdoel_dict["df_cluster"]
    return df_with_initial_clusters


def tree_based_feature_selection(
    df_with_initial_clusters: pd.DataFrame,
    target_based_fs_ranking: pd.DataFrame,
    fs_params: Dict,
    explainer_model_params: Dict,
    importance_model_params: Dict,
) -> pd.DataFrame:
    """Reorder-Supervised FS Ranking into an Unsupervised FS ranking.

    Reorder FS ranking according a feature importance model (Random Forest Optimzed with Bayesian Opt),
    an gives explainability insights though an explainer model (DT Optimzed with Bayesian Opt).

    Args:
        df_with_initial_clusters (pd.DataFrame): Master table with cluster columns
        target_based_fs_ranking (pd.DataFrame): Target based feature selection ranking.
        fs_params (Dict): Feature selection params.
        explainer_model_params (Dict): Explainer model params.
        importance_model_params (Dict): Importance model params.

    Returns:
        pd.DataFrame: Unsupervised Feature Selection Ranking.
    """
    # params
    id_col = fs_params["customer_id_col"]
    # initial receommendation from feature selection
    initial_recommended_features = list(target_based_fs_ranking["feature"].values) + [
        id_col
    ]
    # explain clustering predictions with Bagging or Bossting algorithms + tree plot from a decision tree model
    explainer_dict = clustering_explainer(
        df_with_initial_clusters,
        fs_params,
        initial_recommended_features,
        explainer_model_params,
        importance_model_params,
    )
    # initial feature importance from the explainer
    feature_importance_df = explainer_dict["feature_importance"]

    # recommend features with feature importance greater than a threshold.
    feature_importance_threshold = fs_params["final_ranking"][
        "feature_importance_threshold"
    ]
    recommended_features = feature_importance_df[
        feature_importance_df["feature_importance"] > feature_importance_threshold
    ]["feature_name"].to_list()

    msg1 = f"Features recommended to use on the macro clusters embeddings: {recommended_features}"
    msg2 = f"wich have a feature importance greater than {feature_importance_threshold}"
    logger.warning(msg1 + msg2)
    return feature_importance_df
