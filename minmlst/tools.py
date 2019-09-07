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
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
# region set random seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(c.SEED)
# endregion set random seeds


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


#todo (Isana)- should we return also the number of clusters/the clustering structure it self for a selected num_of_genes + threshold
#todo (Isana)- should we add p.v. to the plot?
def plot_res(analysis_res, measure):
    title = f'Results per number of genes (measure = {measure})'
    x_label = 'Number of genes'
    y_label = 'Adjusted Rand Index  or  p-value'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    x_col = 'num_of_genes'
    y_cols = list(analysis_res.columns.values)
    y_cols.remove(x_col)
    for y_col in y_cols:
        ax.plot(analysis_res[x_col], analysis_res[y_col], label=y_col, marker='o', linestyle='--')
    ax.legend(frameon=True)
    plt.show()


def gene_reduction_analysis(data, gene_importance, measure, reduction=0.2, percentiles=[0.5, 1],
                            find_recommended_thresh=False, simulated_samples=0, plot_results=True, n_jobs=mp.cpu_count()):
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
        validate_input_gra(data, gene_importance, measure, reduction, percentiles, simulated_samples, n_jobs)

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
                delayed(hierarchical_clustering)(ST, X, num_of_genes, gene_importance, percentiles,
                                                 find_recommended_thresh, simulated_samples) for num_of_genes in lst)
        except Exception as ex:
            print(f"Error - unable to perform parallel computing due to: {ex}")
            print(f"Running serial computation instead")
            results = []
            for num_of_genes in lst:
                r = hierarchical_clustering(ST, X, num_of_genes, gene_importance, percentiles, find_recommended_thresh, simulated_samples)
                results = results + [r]

        analysis_res = pd.DataFrame(results)
        analysis_res = reorder_analysis_res(analysis_res)

        if find_recommended_thresh:
            analysis_res = find_threshold(analysis_res, ST, simulated_samples, n_jobs)

        if plot_results:
            plot_res(analysis_res, measure)

        return analysis_res

    except ValueError as ve:
        print(ve)






