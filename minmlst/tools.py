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
from minmlst.clustering import *
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
import os
# todo- add joblib and multiprocessing to imports
from joblib import Parallel, delayed
import multiprocessing as mp
# region set random seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(c.SEED)
# todo-check if random is used
# rn.seed(c.SEED)
# endregion set random seeds


def validate_data(data):
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


def validate_input_gi(data, measures, max_depth, learning_rate, stopping_method, stopping_rounds):
    print("Input validation")
    # data
    validate_data(data)
    # measures
    valid_measures = ['shap', 'weight', 'gain', 'cover', 'total_gain', 'total_cover']
    if not isinstance(measures, (collections.Sequence, np.ndarray)) or len(measures) == 0:
        raise ValueError(f"Error: 'measures' must be a non-empty array. Valid elements are: {valid_measures}.")
    for m in measures:
        if m not in valid_measures:
            raise ValueError(f"Error: 'measures' contains invalid element {m}. Valid elements are: {valid_measures}.")
    # max_depth
    if not np.issubdtype(type(max_depth), np.integer):
        raise ValueError(f"Error: 'max_depth' must be of type int, got {type(max_depth)}")
    # learning_rate
    if not np.issubdtype(type(learning_rate), np.floating):
        raise ValueError(f"Error: 'learning_rate' must be of type float, got {type(learning_rate)}")
    # stopping_method
    if stopping_method not in ['num_boost_round', 'early_stopping_rounds']:
        raise ValueError(f"Error: 'stopping_method' must be 'num_boost_round' or 'early_stopping_rounds' (type str)")
    # stopping_rounds
    if not np.issubdtype(type(stopping_rounds), np.integer):
        raise ValueError(f"Error: 'stopping_rounds' must be of type int, got {type(stopping_rounds)}")


def validate_input_gra(data, gene_importance, measure, reduction, percentiles, n_jobs):
    print("Input validation")
    # data
    validate_data(data)
    # measure
    valid_measures = ['shap', 'weight', 'gain', 'cover', 'total_gain', 'total_cover']
    if measure not in valid_measures:
        raise ValueError(f"Error: measure must be either 'shap', 'weight', 'gain', 'cover', 'total_gain' "
                         f"or 'total_cover'.")
    # gene_importance
    if not isinstance(gene_importance, pd.DataFrame):
        raise ValueError(f"Error: 'gene_importance' must be in the format returned by 'gene_importance' function.")
    gi_cols = gene_importance.columns.values
    if len(gi_cols) < 2 or gi_cols[0] != 'gene':
        raise ValueError(f"Error: 'gene_importance' must be in the format returned by 'gene_importance' function.")
    gi_measures = [col.replace("importance_by_", "") for col in gi_cols[1:]]
    for m in gi_measures:
        if m not in valid_measures:
            raise ValueError(f"Error: 'gene_importance' must be in the format returned by 'gene_importance' function.")
    invalid = [not (np.issubdtype(t, np.number)) for t in gene_importance.dtypes[1:]]
    if sum(invalid) > 0:
        raise ValueError(f"Error: 'gene_importance' contains non-numeric importance scores. "
                         f"Invalid columns and types: \n{[False] + gene_importance.dtypes[invalid]}")
    gi_genes = set(gene_importance.iloc[:, 0])
    data_genes = set(data.columns.values[:-1])
    if len(gi_genes - data_genes) + len(data_genes - gi_genes) > 0:
        raise ValueError(f"Error: genes in 'data' and 'gene_importance' do not match.")
    # measure
    if measure not in gi_measures:
        raise ValueError(f"Error: 'measure' must be included in the 'gene_importance' results -> {gi_measures}.")
    # reduction
    if (not np.issubdtype(type(reduction), np.number)) or (reduction <= 0):
        raise ValueError(f"Error: 'reduction' must be a positive number. Use int for number of genes, or float for"
                         f" percentage of genes to be reduced.")
    # percentiles
    # todo- check if it's a list
    # todo- check if all elements are numbers between 0< to <100
    invalid = [not (np.issubdtype(type(p), np.number)) for p in percentiles]
    if sum(invalid) > 0:
        raise ValueError(f"Error: 'gene_importance' contains non-numeric importance scores. "
                         f"Invalid columns and types: \n{[False] + gene_importance.dtypes[invalid]}")
    # n_jobs
    if not np.issubdtype(type(n_jobs), np.integer):
        raise ValueError(f"Error: 'n_jobs' must be of type int, got {type(n_jobs)}")


def gene_importance(data, measures, max_depth=c.MAX_DEPTH, learning_rate=c.LEARNING_RATE,
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
    #todo- recheck docstrings
    try:
        validate_input_gi(data, measures, max_depth, learning_rate, stopping_method, stopping_rounds)

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


def gene_reduction_analysis(data, gene_importance, measure, reduction=0.2, percentiles=[0.5, 1],
                            find_recommended_thresh=False, simulation=False, plot_results=True, n_jobs=mp.cpu_count()):
    '''

    :param data (DataFrame): (n-1) columns of genes, last column (n) must contain the ST (strain type).
                             Each row represents a profile of a single isolate.
                             Data types should be integers only.
                             Missing values should be represented as 0. No missing values are allowed for the ST.
    :param gene_importance (DataFrame): Gene importance results in the format returned by 'gene_importance' function.
    :param measure (str): The measure according to which gene importance will be defined. measure must be included in
                          the 'gene_importance' results.
                          measure must be either 'shap', 'weight', 'gain', 'cover', 'total_gain' or 'total_cover'.
    :return:
    '''
    try:
        # todo- user to set parameters for h-clustering, monte carlo, threshold selection
        validate_input_gra(data, gene_importance, measure, reduction, percentiles, n_jobs)

        # remove non-informative genes
        gi = gene_importance[gene_importance['importance_by_' + measure] > 0]
        num_informative = len(gi)
        print(f"{num_informative} informative genes were found")
        if reduction < 1:
            reduction = int(num_informative*reduction)
        # todo (Isana)- should we add an iteration with all genes (or show only the informative)?
        # lst = np.arange(num_informative, 0, -reduction)
        lst = [len(gene_importance)] + list(np.arange(num_informative, 0, -reduction))
        # sort importance according to selected measure
        gene_importance = gene_importance.sort_values(by='importance_by_' + measure, ascending=False).reset_index(drop=True)
        X, ST = data.iloc[:, :-1], data.iloc[:, -1]
        print("Hierarchical clustering")
        try:
            results = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
                delayed(hierarchical_clustering)(ST, X, num_of_genes, gene_importance, percentiles, find_recommended_thresh,
                                                 simulation) for num_of_genes in lst)
        except Exception as ex:
            print(f"Error - unable to perform parallel computing due to: {ex}")
            print(f"Running serial computation instead")
            results = []
            for num_of_genes in lst:
                r = hierarchical_clustering(ST, X, num_of_genes, gene_importance, percentiles, find_recommended_thresh, simulation)
                results = results + [r]

        analysis_res = pd.DataFrame(results)
        analysis_res = reorder_analysis_res(analysis_res)

        # todo- plot results
        # if plot_results:
        #     print()

        return analysis_res

    except ValueError as ve:
        print(ve)






