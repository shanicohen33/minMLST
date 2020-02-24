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
import matplotlib.pyplot as plt


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
    if std == 0:
        std = np.finfo(np.float32).eps
    NARI_0 = (ARI_0 - m) / std
    NARI_dist = (ARI_dist - m) / std
    p_value = len(NARI_dist[NARI_dist > NARI_0]) / len(NARI_dist)

    return p_value


def is_next_better(best_ARI_vec, next_ARI_vec):
    minimal_diff = 0
    diff = sum(next_ARI_vec - best_ARI_vec)
    return diff >= minimal_diff


def calc_ARI_pv(res, CT, predicted_CT_lst, percentiles, thresholds, simulated_samples):
    for idx, predicted_CT in enumerate(predicted_CT_lst):
        ARI = adjusted_rand_score(CT, predicted_CT)
        res.update({f"ARI_perc_{format(percentiles[idx], '.15g')}": ARI})
        res.update({f"threshold_value_perc_{format(percentiles[idx], '.15g')}": thresholds[idx]})
        if simulated_samples > 0:
            p_value = simulation_study_ARI(CT, predicted_CT, ARI, simulated_samples)
            res.update({f"pv_perc_{format(percentiles[idx], '.15g')}": p_value})


def calc_ARI_pv_vec(num_of_genes, perc, CT, simulated_samples):
    res = {'num_of_genes': num_of_genes}
    with open(join(c.TEMP_FOLDER, f"thresholds_{num_of_genes}" + '.pickle'), 'rb') as handle:
        thres_per_perc = pickle.load(handle)
    with open(join(c.TEMP_FOLDER, f"z_{num_of_genes}" + '.pickle'), 'rb') as handle:
        z = pickle.load(handle)
    curr_thres = thres_per_perc[perc]
    predicted_CT_lst = [fcluster(Z=z, t=curr_thres, criterion='distance')]
    calc_ARI_pv(res, CT, predicted_CT_lst, [perc], [curr_thres], simulated_samples)

    return res


def parse_ans(ans, perc, simulated_samples):
    ans_df = pd.DataFrame(ans)
    _ans = {"perc": perc, "ARI_vec": ans_df[f"ARI_perc_{format(perc, '.15g')}"]}
    if simulated_samples > 0:
        _ans.update({"pv_vec": ans_df[f"pv_perc_{format(perc, '.15g')}"]})
    return _ans


def find_percentile(results, CT, percentiles_to_check, simulated_samples, n_jobs):
    best_perc = percentiles_to_check[0]
    next_perc = percentiles_to_check[1]
    _best = {"perc": best_perc, "ARI_vec": results[f"ARI_perc_{format(best_perc, '.15g')}"]}
    _next = {"perc": next_perc, "ARI_vec": results[f"ARI_perc_{format(next_perc, '.15g')}"]}
    if simulated_samples > 0:
        _best.update({"pv_vec": results[f"pv_perc_{format(best_perc, '.15g')}"]})
        _next.update({"pv_vec": results[f"pv_perc_{format(next_perc, '.15g')}"]})
    next_better = is_next_better(_best["ARI_vec"], _next["ARI_vec"])

    if len(percentiles_to_check) > 2:
        potential_perc = list(percentiles_to_check[2:])
        while next_better and potential_perc:
            _best = _next
            next_perc = potential_perc.pop(0)
            try:
                ans = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
                    delayed(calc_ARI_pv_vec)(num_of_genes, next_perc, CT, simulated_samples) for num_of_genes in results['num_of_genes'])
            except Exception as ex:
                print(f"Error - unable to perform parallel computing due to: {ex}")
                print(f"Running serial computation instead")
                ans = []
                for num_of_genes in results['num_of_genes']:
                    a = calc_ARI_pv_vec(num_of_genes, next_perc, CT, simulated_samples)
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


def hierarchical_clustering(CT, x, num_of_genes, gene_importance, linkage_method, percentiles, find_percentile,
                            percentiles_to_check, simulated_samples):
    res = {'num_of_genes': num_of_genes}
    curr_genes = gene_importance['gene'][0:num_of_genes]
    curr_x = x.loc[:, curr_genes]
    distances = pdist(X=curr_x, metric=c.DISTANCE_METRIC)
    z = linkage(y=distances, method=linkage_method)

    if find_percentile:
        # save percentiles and thresholds as temp files
        percentiles = percentiles_to_check
        thresholds = np.percentile(a=distances, q=percentiles)
        save_temp_files(percentiles, thresholds, z, num_of_genes)
        # percentiles 0.5 and 1 are calculated as a baseline
        percentiles = percentiles[:2]
        thresholds = thresholds[:2]
    else:
        thresholds = np.percentile(a=distances, q=percentiles)
    predicted_ST_lst = [fcluster(Z=z, t=t, criterion='distance') for t in thresholds]
    calc_ARI_pv(res, CT, predicted_ST_lst, percentiles, thresholds, simulated_samples)

    return res


def reorder_analysis_res(df):
    cols = list(df.columns.values)
    cols.remove('num_of_genes')
    return df[['num_of_genes'] + cols]


def plot_res(analysis_res, measure):
    title = f'Results per number of genes (measure = {measure})'
    x_label = 'Number of genes'
    y_label = 'Adjusted Rand Index  or  p-value'

    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_ylim((-0.05, 1))

    x_col = 'num_of_genes'
    y_cols = list(analysis_res.columns.values)
    y_cols.remove(x_col)
    for y_col in y_cols:
        ax.plot(analysis_res[x_col], analysis_res[y_col], label=y_col, marker='o', linestyle='--')
    ax.legend(frameon=True, bbox_to_anchor=(1, 0.5), loc="center left")
    plt.show()
