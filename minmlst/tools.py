"""
Machine learning module for finding a minimal MLST scheme for bacterial strain typing
=====================================================================================

minMLST is a machine-learning based methodology for identifying a minimal subset of genes that preserves high
discrimination among bacterial strains. It combines well known machine-learning algorithms and approaches such as
XGBoost, distance-based hierarchical clustering, and SHAP.
minMLST quantifies the importance level of each gene in an MLST scheme and allows the user to investigate the trade-off
between minimizing the number of genes in the scheme vs preserving a high resolution among different strain types.

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


def gene_importance(data, measures, max_depth=c.MAX_DEPTH, learning_rate=c.LEARNING_RATE,
                    stopping_method=c.STOPPING_METHOD, stopping_rounds = c.NUM_BOOST_ROUND):
    """
    This function provides a ranking of gene importance according to selected measures: 'shap', 'weight', 'gain',
    'cover', 'total_gain' or 'total_cover'.
        'shap' - the mean magnitude of the SHAP values, i.e. the mean absolute value of the SHAP values of a given gene
                 (See: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions).
        'weight' - the number of times a given gene is used to split the data across all splits.
        'gain' (or 'total gain') - the average (or total) gain is the average (or total) reduction of Multiclass Log Loss
                                   contributed by a given gene across all splits.
        'cover' (or 'total cover') - the average (or total) quantity of observations concerned by a given gene across all splits.
    As a pre-step, STs (strain types) with a single representative isolate are filtered from the dataset.
    Next, an XGBoost model is trained with parameters 'max_depth', 'learning_rate', 'stopping_method' and 'stopping_rounds' -
    more information about XGBoost parameters can be found here: https://xgboost.readthedocs.io/en/latest/python/python_api.html.
    Model's performance is evaluated by Multi-class log loss over a test set.
    Gene importance scores are measured over the trained model and provided as a DataFrame output.

    :param data (DataFrame): DataFrame in the shape of (m,n).
                             (n-1) columns of genes, last column (n) must contain the ST (strain type).
                             Each row (m) represents a profile of a single isolate.
                             Data types should be integers only.
                             Missing values should be represented as 0, no missing values are allowed for the ST (last
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
                            find_recommended_thresh=False, percentiles_to_check = c.PERCENTILES_TO_CHECK,
                            simulated_samples=0, plot_results=True, n_jobs=mp.cpu_count()):
    """
    This function analyzes how minimizing the number of genes in the MLST scheme impacts strain typing performance.
    At each iteration, a subset of X most important genes is selected; and based on the allelic profile composed of these genes,
    isolates are clustered into strain types (ST) using a distance-based hierarchical clustering (complete linkage).
    The distance between every pair of isolates is measured by a normalized Hamming distance, which stands for
    the proportion of those genes between the two allelic profiles which disagree.
    To obtain a clustering structure (i.e. STs), we apply a threshold (or maximal distance between isolates of the same ST)
    that equals to a certain percentile of distances distribution; This percentile (or percentiles) can be defined by
    the user in the 'percentiles' input parameter, or alternatively being selected by the 'find threshold' procedure
    that searches in the search space of 'percentiles_to_check' input parameter (see below).
    Typing performance is measure by the Adjusted Rand index (ARI), which quantifies similarity between the induced
    clustering structure (that is based on a subset of genes) and the original clustering structure (that is based on
    all genes, and was given as an input)(see: https://link.springer.com/article/10.1007/BF01908075).
    p-value for the ARI is calculated using a Monte Carlo simulation study (see: https://www.sciencedirect.com/science/article/abs/pii/S0950329313000852)
    The analysis results are provided as a DataFrame output and are also plotted by default.

    'find threshold' procedure:
    This procedure uses an heuristic to find a suitable threshold for generating an induced clustering structure (i.e. STs).
    The search space is the list of percentiles ('percentiles_to_check') provided as an input parameter.
    To represent the typing performance achieved by a particular threshold, we compute the ARI it results for each
    subset of selected genes and construct a vector composed of these ARI elements.
    To find a potentially more precise threshold, we runs a serial evaluation process starting from the minimal
    threshold (baseline) up to the maximal threshold in the search space.
    At each iteration, the ARI vector of the 'baseline' threshold is compared with the ARI vector of the 'next'
    (second minimal) threshold using a distance function; this function is a "non-absolute" L1 distance, i.e. it equals
    to the sum of the differences of two vectors' coordinates, when subtracting the 'baseline' from the 'next'.
    In case the distance is positive (i.e. 'next' performs better) the 'next' threshold will be defined as the new
    'baseline' and will be compared with the next potential threshold. otherwise, the search is done and the 'baseline'
    is selected as the recommended threshold.


    :param data (DataFrame): DataFrame in the shape of (m,n).
                         (n-1) columns of genes, last column (n) must contain the ST (strain type).
                         Each row (m) represents a profile of a single isolate.
                         Data types should be integers only.
                         Missing values should be represented as 0, no missing values are allowed for the ST (last
                         column).
    :param gene_importance (DataFrame): Gene importance results in the format returned by 'gene_importance' function.
    :param measure (str): The measure according to which gene importance will be defined. measure must be included in
                      the 'gene_importance' results.
                      measure must be either 'shap', 'weight', 'gain', 'cover', 'total_gain' or 'total_cover'.
    :param reduction (numeric): The number (int) or percentage (0<float<1) of least important genes to be removed at
                                each iteration (default = 0.2). Importance is determined by parameter 'measure'.
                                First iteration includes all genes, second iteration includes all informative genes
                                (genes with importance score > 0), and the subset of genes in the subsequent iterations
                                is calculated according to the 'reduction' parameter.
    :param percentiles (float or array of floats): The percentile (or percentiles) of distances distribution to be used
                                                   as a threshold (or thresholds) -> (default = [0.5, 1]).
                                                   Each percentile must be greater than 0 and smaller than 100.
                                                   The threshold defines the maximal distance (or dissimilarity) between
                                                   isolates of the same ST (cluster).
    :param find_recommended_thresh (boolean): if True, ignore parameter 'percentiles' and run a procedure to find a
                                              recommended threshold out of 'percentiles_to_check' (default = False).
    :param percentiles_to_check (array of floats): The percentiles of distances distribution to be evaluated and compared,
                                                   in case parameter 'find_recommended_thresh' is True (default = [0.5, 1, ... ,20]).
                                                   Array must contain at least 2 percentiles, each percentile must be
                                                   greater than 0 and smaller than 100.
    :param simulated_samples (int): The number of samples (clustering structures) to simulate for computing the p-value
                                    of the observed ARI (default = 0).
                                    In case 'simulated_samples'=0, simulation won't run and p-value won't be calculated.
                                    For the significance of the results, it's recommended to use at least ~1000 samples
                                    (see- https://www.sciencedirect.com/science/article/pii/S0950329313000852).
    :param plot_results (boolean): if True, plot the ARI and p-value results for each selected threshold (parameter 'percentiles')
                                   as a function of the number of genes (default = True).
    :param n_jobs (int): The maximum number of concurrently running jobs.
                         (default = min(60, number of CPUs in the system)).
    :return: DataFrame. ARI and p-value (if 'simulated_samples' > 0) computed for each subset of most important genes,
                        by using a selected threshold (parameter 'percentiles').
    """
    try:
        n_jobs, percentiles_to_check = validate_input_gra(data, gene_importance, measure, reduction, percentiles,
                                                          percentiles_to_check, simulated_samples, n_jobs)
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
        X, ST = data.iloc[:, :-1], data.iloc[:, -1]

        print("Hierarchical clustering")
        try:
            results = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
                delayed(hierarchical_clustering)(ST, X, num_of_genes, gene_importance, percentiles,
                                                 find_recommended_thresh, percentiles_to_check, simulated_samples)
                for num_of_genes in lst)
        except Exception as ex:
            print(f"Error - unable to perform parallel computing due to: {ex}")
            print(traceback.format_exc())
            print(f"Running serial computation instead")
            results = []
            for num_of_genes in lst:
                r = hierarchical_clustering(ST, X, num_of_genes, gene_importance, percentiles, find_recommended_thresh,
                                            percentiles_to_check, simulated_samples)
                results = results + [r]

        analysis_res = pd.DataFrame(results)
        analysis_res = reorder_analysis_res(analysis_res)

        if find_recommended_thresh:
            analysis_res = find_threshold(analysis_res, ST, percentiles_to_check, simulated_samples, n_jobs)

        if plot_results:
            plot_res(analysis_res, measure)

        return analysis_res

    except ValueError as ve:
        print(ve)






