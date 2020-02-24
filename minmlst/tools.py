"""
Machine learning module for finding a minimal MLST scheme for bacterial strain typing
=====================================================================================

minMLST is a machine-learning based methodology for identifying a minimal subset of genes that preserves high
discrimination among bacterial strains. It combines well known machine-learning algorithms and approaches such as
XGBoost, distance-based hierarchical clustering, and SHAP.
minMLST quantifies the importance level of each gene in an MLST scheme and allows the user to investigate the trade-off
between minimizing the number of genes in the scheme vs preserving a high resolution among strains.

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
from minmlst.input_validation import *
import pandas as pd
import numpy as np
import os
import traceback
from joblib import Parallel, delayed
import multiprocessing as mp
# region set random seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(c.SEED)
# endregion set random seeds


def gene_importance(data, measures, max_depth=6, learning_rate=0.3, stopping_method='num_boost_round',
                    stopping_rounds = 100):
    """
    This function provides a ranking of gene importance according to selected measures: 'shap', 'weight', 'gain',
    'cover', 'total_gain' or 'total_cover'.
        'shap' - the mean magnitude of the SHAP values, i.e. the mean absolute value of the SHAP values of a given gene
                 (See: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions).
        'weight' - the number of times a given gene is used to split the data across all splits.
        'gain' (or 'total gain') - the average (or total) gain is the average (or total) reduction of Multiclass Log Loss
                                   contributed by a given gene across all splits.
        'cover' (or 'total cover') - the average (or total) quantity of observations concerned by a given gene across all splits.

    As a pre-step, CTs (cluster types) with a single representative isolate are filtered from the dataset.
    Next, an XGBoost model is trained with parameters 'max_depth', 'learning_rate', 'stopping_method' and 'stopping_rounds' -
    more information about XGBoost parameters can be found here: https://xgboost.readthedocs.io/en/latest/python/python_api.html.
    Model's performance is evaluated by Multi-class log loss over a test set.
    Finally, gene importance values are measured for the trained model and provided as a DataFrame output.

    :param data (DataFrame): DataFrame in the shape of (m,n).
                             (n-1) columns of genes, last column (n) must contain the CT (cluster type).
                             Each row (m) represents a profile of a single isolate.
                             Data types should be integers only.
                             Missing values should be represented as 0, no missing values are allowed for the CT (last
                             column).
    :param measures (array): an array containing at least one of the following measures (str type):
                            ['shap', 'weight', 'gain', 'cover', 'total_gain', 'total_cover'].
    :param max_depth (int): Maximum tree depth for base learners (default = 6). Must be greater equal to 0.
    :param learning_rate (float): Boosting learning rate (default = 0.3). Must be greater equal to 0.
    :param stopping_method (str): 'num_boost_round' or 'early_stopping_rounds' (default = 'num_boost_round').
    :param stopping_rounds (int): Number of rounds for boosting or early stopping (default = 100).
                                  Must be greater than 0.
    :return: DataFrame. Importance score per gene according to each of the input measures.
                        Higher scores are given to more important (informative) genes.
    """

    try:
        validate_input_gi(data, measures, max_depth, learning_rate, stopping_method, stopping_rounds)

        # Filter singletons (strain types with a single related isolate)
        st_lst = data.iloc[:, -1]
        st, counts = np.unique(st_lst, return_counts=True)
        data = data[[x in st[counts > 1] for x in st_lst]].reset_index(drop=True)
        if data.empty:
            raise ValueError(f"Error: 'data' contains only singletons (strain types with a single related isolate).")

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        results = get_gene_importance(X, y, measures, max_depth, learning_rate, stopping_method, stopping_rounds)
        return results

    except ValueError as ve:
        print(ve)


def gene_reduction_analysis(data, gene_importance, measure, reduction=0.2, linkage_method='complete',
                            percentiles=[0.5, 1], find_recommended_percentile=False,
                            percentiles_to_check = np.arange(.5, 20.5, 0.5), simulated_samples=0, plot_results=True,
                            n_jobs=mp.cpu_count()):
    """
    This function analyzes how minimizing the number of genes in the MLST scheme impacts strain typing performance.
    At each iteration, a reduced subset of most important genes is selected; and based on the allelic profile composed of these genes,
    isolates are clustered into cluster types (CTs) using a distance-based hierarchical clustering.
    The distance between every pair of isolates is measured by a normalized Hamming distance, which quantifies
    the proportion of the genes which disagree on their allelic assignment. The distance between any two clusters
    is determined by the 'linkage' input parameter.

    To obtain the partition into strains (i.e. the induced CTs), we apply a threshold on the distance between isolates
    belonging to the same cluster-type. we use percentile-based thresholds instead of constant thresholds, i.e. for a
    given percentile of distances' distribution, we dynamically calculate the threshold value for each subset of genes.
    The percentile (or percentiles) can be defined by the user in the `percentiles` input parameter, or alternatively
    being selected by the **<em>find recommended percentile</em>** procedure (depicted below) that searches in the
    search space of `percentiles_to_check` input parameter.

    Typing performance is measure by the Adjusted Rand Index (ARI), which quantifies similarity between the induced
    partition into cluster types (that is based on a subset of genes) and the original partition in cluster types (that is based on
    all genes, and was given as an input) (see: https://link.springer.com/article/10.1007/BF01908075).
    p-value for the ARI is calculated using a Monte Carlo simulation study (see: https://www.sciencedirect.com/science/article/abs/pii/S0950329313000852)
    The analysis results are provided as a DataFrame output and are also plotted by default.

    'find recommended percentile' procedure:
    This procedure uses an heuristic to find a recommended percentile. This is the percentile
    with the best overall predictive performance, which is equivalent to the ARI curve with the highest AUC, and is
    referred as ‘best’. At first, we initialize ‘best’ to the minimal percentile in search space (`percentiles_to_check`).
    Then we compare ‘best’ to the successor percentile in `percentiles_to_check`, referred as ‘next’,
    by computing the "non-absolute" L1 distance between their ARI vectors.
    This distance equals to the sum of the differences between the two vectors when subtracting the 'best' from the 'next'.
    In case the distance is not negative (i.e., 'next' performs better or the same), 'next' is defined as the new 'best'.
    Otherwise, the search is completed and 'best' is selected as the recommended percentile.



    :param data (DataFrame): DataFrame in the shape of (m,n).
                         (n-1) columns of genes, last column (n) must contain the CT (cluster type).
                         Each row (m) represents a profile of a single isolate.
                         Data types should be integers only.
                         Missing values should be represented as 0, no missing values are allowed for the CT (last
                         column).
    :param gene_importance (DataFrame): Importance scores in the format returned by 'gene_importance' function.
    :param measure (str): A single measure according to which gene importance will be defined.
                          Can be either 'shap', 'weight', 'gain', 'cover', 'total_gain' or 'total_cover'.
                          Note that the selected measure must be included in the `gene_importance` input
    :param reduction (numeric): The number (int) or percentage (0<float<1) of least important genes to be removed at
                                each iteration (default = 0.2).
                                The first iteration includes all genes, the second iteration includes all informative
                                genes (importance score > 0), and the subsequent iterations include a reduced
                                subset of genes according to the `reduction` parameter
    :param linkage_method (str): The linkage method to compute the distance between clusters in the hierarchical
                                 clustering algorithm (default = 'complete').
                                 Can be either 'single', 'complete', 'average', 'weighted', 'centroid', 'median' or
                                 'ward'.
                                 for more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    :param percentiles (float or array of floats): The percentile (or percentiles) of distances distribution to be used
                                                   (default = [0.5, 1]).
                                                   Each percentile must be greater than 0 and smaller than 100.
                                                   For a given percentile, we dynamically calculate the threshold value
                                                   for each subset of genes.
                                                   The threshold value refers to the distance between isolates of the
                                                   same CT, which is defined as the proportion of genes that disagree on
                                                   their allelic assignment.
    :param find_recommended_percentile (boolean): if True, ignore parameter 'percentiles' and run a procedure to find a
                                                  recommended threshold out of 'percentiles_to_check' (default = False).
                                                  The outputs will be the ARI and p-value results computed for each
                                                  subset of genes when using the recommended percentile.
    :param percentiles_to_check (array of floats): The percentiles of distances distribution to be evaluated by the
                                                   'find recommended percentile' procedure (default = numpy.arange(.5, 20.5, 0.5)).
                                                   The array must contain at least 2 percentiles; each percentile must
                                                   be greater than 0 and smaller than 100.
    :param simulated_samples (int): The number of samples (partitions) to simulate the computation of the
                                    p-value of the observed ARI (default = 0).
                                    For the significance of the p-value results, it's recommended to use ~1000 samples
                                    or more (see- https://www.sciencedirect.com/science/article/pii/S0950329313000852).
                                    In case 'simulated_samples'=0, simulation won't run and p-value won't be calculated.
    :param plot_results (boolean): if True, plot the ARI and p-value results for each selected percentile
                                   as a function of the number of genes (default = True).
    :param n_jobs (int): The maximum number of concurrently running jobs.
                         (default = min(60, number of CPUs in the system)).
    :return: DataFrame. ARI and p-value (if 'simulated_samples' > 0) computed for each subset of most important genes,
                        and for each selected percentile.
    """
    try:
        n_jobs, percentiles_to_check = validate_input_gra(data, gene_importance, measure, reduction, linkage_method,
                                                          percentiles, percentiles_to_check, simulated_samples, n_jobs)
        # remove non-informative genes
        gi = gene_importance[gene_importance['importance_by_' + measure] > 0]
        num_informative = len(gi)
        print(f"{num_informative} informative genes were found")
        if reduction < 1:
            reduction = int(num_informative*reduction)
        # lst = np.arange(num_informative, 0, -reduction)
        lst = [len(gene_importance)] + list(np.arange(num_informative, 0, -reduction))
        # sort importance according to selected measure
        gene_importance = gene_importance.sort_values(by='importance_by_' + measure, ascending=False).reset_index(drop=True)
        X, CT = data.iloc[:, :-1], data.iloc[:, -1]

        print("Hierarchical clustering")
        try:
            results = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
                delayed(hierarchical_clustering)(CT, X, num_of_genes, gene_importance, linkage_method, percentiles,
                                                 find_recommended_percentile, percentiles_to_check, simulated_samples)
                for num_of_genes in lst)
        except Exception as ex:
            print(f"Error - unable to perform parallel computing due to: {ex}")
            print(traceback.format_exc())
            print(f"Running serial computation instead")
            results = []
            for num_of_genes in lst:
                r = hierarchical_clustering(CT, X, num_of_genes, gene_importance, linkage_method, percentiles,
                                            find_recommended_percentile, percentiles_to_check, simulated_samples)
                results = results + [r]

        analysis_res = pd.DataFrame(results)

        if find_recommended_percentile:
            analysis_res = find_percentile(analysis_res, CT, percentiles_to_check, simulated_samples, n_jobs)
        analysis_res = reorder_analysis_res(analysis_res)
        f_analysis_res = analysis_res.loc[:, ~analysis_res.columns.str.startswith("threshold")]

        if plot_results:
            plot_res(f_analysis_res, measure)

        return f_analysis_res

    except ValueError as ve:
        print(ve)






