import minmlst.config as c
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from numpy.random import permutation
import time
import pandas as pd
import pickle
import os
from os.path import join, exists
from minmlst.utils import *


def save_temp_files(percentiles, thresholds, z, num_of_genes):
    # todo- make sure parallel can work with this (cosider removing to outer functions + delete of this folder)
    print(c.TEMP_FOLDER)
    create_dir_if_not_exists(c.TEMP_FOLDER)

    thres_per_perc = dict(zip(percentiles, thresholds))
    with open(join(c.TEMP_FOLDER, f"thresholds_{num_of_genes}" + '.pickle'), 'wb') as handle:
        pickle.dump(thres_per_perc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(c.TEMP_FOLDER, f"z_{num_of_genes}" + '.pickle'), 'wb') as handle:
        pickle.dump(z, handle, protocol=pickle.HIGHEST_PROTOCOL)


def is_next_better(best, next):
    #todo- check when next is worst (should "sum" both better and worst)
    #todo- check in "how much" is it better or worst
    # sum_better = sum(np.array(best) < np.array(next))
    # if sum_better > 0:
    #     return True
    #todo-change to False
    return True


def calc_ARI_pv_vec(perc, simulated_samples):
    print("clac ARI")
    res = {"perc": perc, "ARI_vec": perc}
    if simulated_samples > 0:
        print("clac also pv")
        res.update({"pv_vec": perc})
    return res


# todo- at each iteration check if the next result (ARI line) is better than the previous.
#  if next is better, make it the basline and check the next 'next'. if not, return the basline (ari +pv) of the baseline
#  in the format of "results" sent to the function.
# todo- for each number of genes we already have the dendogram, we just need to create clustering and clac ari +pv (if asked)
#  it should be an ITERATIVE procees over the potetial percentails. per potential percetile - PARALLEL to calc ari + ov per num of genes.
#  maybe call h clustering with a certain potential percentaile and join its output
def find_threshold(results, simulated_samples):
    best_perc = c.PERCENTILES_TO_CHECK[0]
    next_perc = c.PERCENTILES_TO_CHECK[1]
    _best = {"perc": best_perc, "ARI_vec": results[f"ARI_perc_{format(best_perc, '.15g')}"]}
    _next = {"perc": next_perc, "ARI_vec": results[f"ARI_perc_{format(next_perc, '.15g')}"]}
    if simulated_samples > 0:
        _best.update({"pv_vec": results[f"pv_perc_{format(best_perc, '.15g')}"]})
        _next.update({"pv_vec": results[f"pv_perc_{format(next_perc, '.15g')}"]})
    # todo- implement is_next_better
    next_better = is_next_better(_best["ARI_vec"], _next["ARI_vec"])

    potential_perc = list(c.PERCENTILES_TO_CHECK[2:])
    while next_better and potential_perc:
        _best = _next
        next_perc = potential_perc.pop(0)
        #todo- implement calc_ARI_pv_vec (parallel per num of genes)
        _next = calc_ARI_pv_vec(next_perc, simulated_samples)
        next_better = is_next_better(_best["ARI_vec"], _next["ARI_vec"])
        print(f"_best: {_best}, _next:{_next}")
    if next_better:
        _best = _next

    res = {'num_of_genes': results['num_of_genes'], f"ARI_perc_{format(_best['perc'], '.15g')}": _best["ARI_vec"]}
    if simulated_samples > 0:
        res.update({f"pv_perc_{format(_best['perc'], '.15g')}": _best["pv_vec"]})

    return pd.DataFrame(res)


def simulation_study_ARI(partition_A, partition_B, ARI_0, num_of_samples):
    ARI_dist = np.empty(num_of_samples, dtype=float)
    for i in range(num_of_samples):
        ARI_dist[i] = adjusted_rand_score(permutation(partition_A), permutation(partition_B))
    m = np.average(ARI_dist)
    std = np.std(ARI_dist)
    NARI_0 = (ARI_0 - m) / std
    NARI_dist = (ARI_dist - m) / std
    p_value = len(NARI_dist[NARI_dist > NARI_0]) / len(NARI_dist)

    return p_value


def hierarchical_clustering(ST, X, num_of_genes, gene_importance, percentiles, find_thresh, simulated_samples):
    print(f"num_of_genes: {num_of_genes}")
    res = {'num_of_genes': num_of_genes}
    curr_genes = gene_importance['gene'][0:num_of_genes]
    curr_X = X.loc[:, curr_genes]

    start1 = time.time()
    # given X with x number of genes
    distances = pdist(X=curr_X, metric=c.DISTANCE_METRIC)
    print(f"elapsed time distances: {time.time() - start1}")

    start2 = time.time()
    z = linkage(y=distances, method=c.HC_METHOD)
    print(f"elapsed time linkage: {time.time() - start2}")

    # in case we need to find the recommended thresh, reset percentiles
    # todo- finish implementation for finding threshold
    if find_thresh:
        start222 = time.time()
        percentiles = c.PERCENTILES_TO_CHECK
        thresholds = np.percentile(a=distances, q=percentiles)
        print(f"elapsed time percentile: {time.time() - start222}")
        save_temp_files(percentiles, thresholds, z, num_of_genes)

        # In case 'find_recommended_thresh' = True ---> percentiles 0.5 and 1 are calculated as a baseline
        percentiles = percentiles[:2]
        predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds[:2]]
    else:
        thresholds = np.percentile(a=distances, q=percentiles)
        predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds]

    for idx, predicted_ST in enumerate(predicted_ST_lst):
        ARI = adjusted_rand_score(ST, predicted_ST)
        res.update({f"ARI_perc_{format(percentiles[idx], '.15g')}": ARI})
        if simulated_samples > 0:
            p_value = simulation_study_ARI(ST, predicted_ST, ARI, simulated_samples)
            res.update({f"pv_perc_{format(percentiles[idx], '.15g')}": p_value})

    return res


def reorder_analysis_res(df):
    cols = list(df.columns.values)
    cols.remove('num_of_genes')
    return df[['num_of_genes'] + cols]


# def calc_ARI(calc_pv):
#     percentile =
#     distances=
#     z =
#     max_distance = np.percentile(a=distances, q=percentile)
#     predicted_ST = fcluster(Z=z, t=max_distance, criterion='distance')
#     cgMLST = #partition_A
#     partition_A = np.array(cgMLST)
#     partition_B = predicted_ST
#     ARI_0 = adjusted_rand_score(partition_A, partition_B)
#     if calc_pv:
#         p_value = simulation_study_ARI(partition_A, partition_B, ARI_0)
#
#     return p_value
