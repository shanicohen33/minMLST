import minmlst.config as c
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from numpy.random import permutation
from joblib import Parallel, delayed
import pandas as pd
import pickle
from os.path import join
from minmlst.utils import *


def save_temp_files(percentiles, thresholds, z, num_of_genes):
    create_dir_if_not_exists(c.TEMP_FOLDER)
    thres_per_perc = dict(zip(percentiles, thresholds))
    with open(join(c.TEMP_FOLDER, f"thresholds_{num_of_genes}" + '.pickle'), 'wb') as handle:
        pickle.dump(thres_per_perc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(c.TEMP_FOLDER, f"z_{num_of_genes}" + '.pickle'), 'wb') as handle:
        pickle.dump(z, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def is_next_better(best_ARI_vec, next_ARI_vec):
    #todo (Isana)- should we let the user define it?
    minimal_diff = 0
    diff = sum(next_ARI_vec - best_ARI_vec)
    return diff >= minimal_diff


def calc_ARI_pv(res, ST, predicted_ST_lst, percentiles, simulated_samples):
    for idx, predicted_ST in enumerate(predicted_ST_lst):
        ARI = adjusted_rand_score(ST, predicted_ST)
        res.update({f"ARI_perc_{format(percentiles[idx], '.15g')}": ARI})
        if simulated_samples > 0:
            p_value = simulation_study_ARI(ST, predicted_ST, ARI, simulated_samples)
            res.update({f"pv_perc_{format(percentiles[idx], '.15g')}": p_value})


def calc_ARI_pv_vec(num_of_genes, perc, ST, simulated_samples):
    res = {'num_of_genes': num_of_genes}
    with open(join(c.TEMP_FOLDER, f"thresholds_{num_of_genes}" + '.pickle'), 'rb') as handle:
        thres_per_perc = pickle.load(handle)
    with open(join(c.TEMP_FOLDER, f"z_{num_of_genes}" + '.pickle'), 'rb') as handle:
        z = pickle.load(handle)
    curr_thres = thres_per_perc[perc]
    predicted_ST_lst = [fcluster(Z=z, t=curr_thres, criterion='distance')]

    calc_ARI_pv(res, ST, predicted_ST_lst, [perc], simulated_samples)

    return res


def parse_ans(ans, perc, simulated_samples):
    ans_df = pd.DataFrame(ans)
    _ans = {"perc": perc, "ARI_vec": ans_df[f"ARI_perc_{format(perc, '.15g')}"]}
    if simulated_samples > 0:
        _ans.update({"pv_vec": ans_df[f"pv_perc_{format(perc, '.15g')}"]})
    return _ans


def find_threshold(results, ST, simulated_samples, n_jobs):
    best_perc = c.PERCENTILES_TO_CHECK[0]
    next_perc = c.PERCENTILES_TO_CHECK[1]
    _best = {"perc": best_perc, "ARI_vec": results[f"ARI_perc_{format(best_perc, '.15g')}"]}
    _next = {"perc": next_perc, "ARI_vec": results[f"ARI_perc_{format(next_perc, '.15g')}"]}
    if simulated_samples > 0:
        _best.update({"pv_vec": results[f"pv_perc_{format(best_perc, '.15g')}"]})
        _next.update({"pv_vec": results[f"pv_perc_{format(next_perc, '.15g')}"]})
    next_better = is_next_better(_best["ARI_vec"], _next["ARI_vec"])

    potential_perc = list(c.PERCENTILES_TO_CHECK[2:])
    while next_better and potential_perc:
        _best = _next
        next_perc = potential_perc.pop(0)
        try:
            ans = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
                delayed(calc_ARI_pv_vec)(num_of_genes, next_perc, ST, simulated_samples) for num_of_genes in results['num_of_genes'])
        except Exception as ex:
            print(f"Error - unable to perform parallel computing due to: {ex}")
            print(f"Running serial computation instead")
            ans = []
            for num_of_genes in results['num_of_genes']:
                a = calc_ARI_pv_vec(num_of_genes, next_perc, ST, simulated_samples)
                ans = ans + [a]
        _next = parse_ans(ans, next_perc, simulated_samples)
        next_better = is_next_better(_best["ARI_vec"], _next["ARI_vec"])
    if next_better:
        _best = _next

    remove_dir(c.TEMP_FOLDER)
    res = {'num_of_genes': results['num_of_genes'], f"ARI_perc_{format(_best['perc'], '.15g')}": _best["ARI_vec"]}
    if simulated_samples > 0:
        res.update({f"pv_perc_{format(_best['perc'], '.15g')}": _best["pv_vec"]})

    return pd.DataFrame(res)


def hierarchical_clustering(ST, x, num_of_genes, gene_importance, percentiles, find_thresh, simulated_samples):
    res = {'num_of_genes': num_of_genes}
    curr_genes = gene_importance['gene'][0:num_of_genes]
    curr_x = x.loc[:, curr_genes]
    distances = pdist(X=curr_x, metric=c.DISTANCE_METRIC)
    z = linkage(y=distances, method=c.HC_METHOD)

    if find_thresh:
        percentiles = c.PERCENTILES_TO_CHECK
        thresholds = np.percentile(a=distances, q=percentiles)
        save_temp_files(percentiles, thresholds, z, num_of_genes)
        # In case 'find_recommended_thresh' = True ---> percentiles 0.5 and 1 are calculated as a baseline
        percentiles = percentiles[:2]
        predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds[:2]]
    else:
        thresholds = np.percentile(a=distances, q=percentiles)
        predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds]
    calc_ARI_pv(res, ST, predicted_ST_lst, percentiles, simulated_samples)

    return res


def reorder_analysis_res(df):
    cols = list(df.columns.values)
    cols.remove('num_of_genes')
    return df[['num_of_genes'] + cols]
