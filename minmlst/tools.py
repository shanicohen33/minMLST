"""
Machine learning module for finding a minimal MLST scheme for bacterial strain typing
=====================================================================================

TBD
"""
# Let users know if they're missing any of the hard dependencies
hard_dependencies = ("shap", "xgboost", "sklearn")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies

from minmlst.gene_importance import *
from minmlst.tests import *
import shap
import xgboost as xgb
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import collections
import time
from os.path import join
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np


def test_func_double(x):
    print('XGBClassifier start')
    model = XGBClassifier(random_state=7, nthread=-1, importance_type='gain')
    print('XGBClassifier end')

    print('xgb.DMatrix start')
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([1, 1, 0])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    print('xgb.DMatrix end')

    return 2*x


def gene_importance(data: pd.DataFrame, measures, max_depth=c.MAX_DEPTH, learning_rate=c.LEARNING_RATE,
                    stopping_method=c.STOPPING_METHOD, stopping_rounds = c.NUM_BOOST_ROUND):
    '''

    :param data (DataFrame): (n-1) columns of genes, last column (n) must contain the ST (strain type).
                             Each row represents a profile of a single isolate.
                             Data types should be integers only.
                             Missing values should be represented as 0. No missing values are allowed for the ST.
    :param measures (array): an array containing at least one of the following measures (str type):
                            ['shap', 'weight', 'gain', 'cover', 'total_gain', 'total_cover'].
    :param max_depth (int): Maximum tree depth for base learners (default = 6).
    :param learning_rate (float): Boosting learning rate (default = 0.3).
    :param stopping_method (str): 'num_boost_round' or 'early_stopping_rounds' (default = 'num_boost_round').
    :param stopping_rounds (int): Number of rounds for boosting or early stopping (default = 100).
    :return:
    '''
    try:
        print("Input validation")
        # data
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Error: 'data' must be of type <class 'pandas.core.frame.DataFrame'>,"
                             f" got {type(data)}.")
        elif data.empty:
            raise ValueError(f"Error: 'data' is empty.")
        invalid = [not (np.issubdtype(t, np.integer)) for t in data.dtypes]
        if sum(invalid) > 0:
            raise ValueError(f"Error: 'data' contains non-integer elements. Invalid columns and types:"
                             f"\n{data.dtypes[invalid]}")
        if 0 in list(data.iloc[:, -1]):
            raise ValueError(f"Error: strain-type column (last) contains missing values, i.e value = 0")
        # measures
        valid_measures = ['shap', 'weight', 'gain', 'cover', 'total_gain', 'total_cover']
        if not isinstance(measures, (collections.Sequence, np.ndarray)) or len(measures) == 0:
            raise ValueError(f"Error: 'measures' must be a non-empty array. Valid elements are: {valid_measures}.")
        # measures
        for m in measures:
            if m not in valid_measures:
                raise ValueError(f"Error: 'measures' contains invalid element {m}. Valid elements are: {valid_measures}.")
        # max_depth
        if not np.issubdtype(type(max_depth), np.integer):
            raise ValueError(
                f"Error: 'max_depth' must be of type int, got {type(max_depth)}")
        # learning_rate
        if not isinstance(learning_rate, float):
            raise ValueError(
                f"Error: 'learning_rate' must be of type float, got {type(learning_rate)}")
        # stopping_method
        if stopping_method not in ['num_boost_round', 'early_stopping_rounds']:
            raise ValueError(f"Error: 'stopping_method' must be 'num_boost_round' or 'early_stopping_rounds' (type str)")
        # stopping_rounds
        if not np.issubdtype(type(stopping_rounds), np.integer):
            raise ValueError(f"Error: 'stopping_rounds' must be of type int, got {type(stopping_rounds)}")

        print(f"Fliter singletones (strain-types with a single isolate only)")
        st_lst = data.iloc[:, -1]
        st, counts = np.unique(st_lst, return_counts=True)
        data = data[[x in st[counts > 1] for x in st_lst]].reset_index(drop=True)
        print(f"   {len(data)} isolates remained out of {len(st_lst)}")

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        results = get_gene_importance(X, y, measures, max_depth, learning_rate, stopping_method, stopping_rounds)
        return results

    except ValueError as ve:
        print(ve)


def gene_reduction_analysis(data, gene_importance, measure, results_path=None):
    # todo-define the format for the input (separate to X and ST)
    # todo- complete missing values
    # todo-check path is correct
    # todo- define a unique name for results file
    # todo- user to set parameters for h-clustering, monte carlo, threshold selection
    # todo - add parallel computation
    # todo- sort again (with ascending=False?) before clustering

    return




