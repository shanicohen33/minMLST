import minmlst.config as c
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from numpy.random import permutation
import time
# todo- make sure pickle is installed
import pickle


def save_temp_files(percentiles, thresholds, z):
    thres_per_perc = dict(zip(percentiles, thresholds))

    # save z thres_per_perc
    # todo- change name (according to number of genes)
    start21 = time.time()
    with open('thres_per_perc' '.pickle', 'wb') as handle:
        pickle.dump(thres_per_perc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"elapsed time linkage.dump: {time.time() - start21}")

    # save z param
    # todo- change name (according to number of genes)
    start21 = time.time()
    with open('z' '.pickle', 'wb') as handle:
        pickle.dump(z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"elapsed time linkage.dump: {time.time() - start21}")


def hierarchical_clustering(ST, X, num_of_genes, gene_importance, percentiles, find_thresh, simulation):
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
        percentiles = np.arange(.5, c.MAX_PERCENTILE + 0.5, 0.5)
        thresholds = np.percentile(a=distances, q=percentiles)
        print(f"elapsed time percentile: {time.time() - start222}")
        save_temp_files(percentiles, thresholds, z)

        # In case 'find_recommended_thresh' = True ---> percentiles 0.5 and 1 are calculated as a baseline
        baseline_idx = 2
        percentiles = percentiles[:baseline_idx]
        predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds[:baseline_idx]]
    else:
        thresholds = np.percentile(a=distances, q=percentiles)
        predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds]

    for idx, predicted_ST in enumerate(predicted_ST_lst):
        ARI = adjusted_rand_score(ST, predicted_ST)
        res.update({f"ARI_prec_{percentiles[idx]}": ARI})
        if simulation:
            p_value = simulation_study_ARI(ST, predicted_ST, ARI)
            res.update({f"pv_prec_{percentiles[idx]}": p_value})

    return res


def simulation_study_ARI(partition_A, partition_B, ARI_0):
    ARI_dist = np.empty(c.SIMULATION_NUM_OF_SAMPLES, dtype=float)
    for i in range(c.SIMULATION_NUM_OF_SAMPLES):
        ARI_dist[i] = adjusted_rand_score(permutation(partition_A), permutation(partition_B))
    m = np.average(ARI_dist)
    std = np.std(ARI_dist)
    NARI_0 = (ARI_0 - m) / std
    NARI_dist = (ARI_dist - m) / std
    p_value = len(NARI_dist[NARI_dist > NARI_0]) / len(NARI_dist)

    return p_value


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
