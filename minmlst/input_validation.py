import pandas as pd
import numpy as np
import collections


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


def validate_percentiles(percentiles):
    valid = False
    if np.issubdtype(type(percentiles), np.number) and 0 < percentiles < 100:
        valid = True
    if isinstance(percentiles, (collections.Sequence, np.ndarray)) and len(percentiles) > 0:
        for p in percentiles:
            if (not np.issubdtype(type(p), np.number)) or (not(0 < p < 100)):
                valid = False
                break
            else:
                valid = True
    if not valid:
        raise ValueError(f"Error: 'percentiles' must be a number or an array of numbers, "
                         f"as each number N is 0 < N < 100.")


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


def validate_input_gra(data, gene_importance, measure, reduction, percentiles, simulation_iter, n_jobs):
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
    validate_percentiles(percentiles)
    # simulation_iter
    if not np.issubdtype(type(simulation_iter), np.integer):
        raise ValueError(f"Error: 'simulation_iter' must be of type int, got {type(simulation_iter)}")
    if simulation_iter < 1000:
        print("Warning: for the significance of the p.v results, a larger number of simulated samples (1000, say) is "
              "required. \nSee- https://www.sciencedirect.com/science/article/pii/S0950329313000852\n")
    # n_jobs
    if not np.issubdtype(type(n_jobs), np.integer):
        raise ValueError(f"Error: 'n_jobs' must be of type int, got {type(n_jobs)}")
    return min(n_jobs, 60)  # limit n_jobs to 60
