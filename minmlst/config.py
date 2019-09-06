import multiprocessing as mp
import os
from os.path import join
import numpy as np

SEED = 10

# region params gene importance
OBJECTIVE = 'multi:softmax'
MAX_DEPTH = 6
LEARNING_RATE = 0.3
EVAL_METRIC = ['merror', 'mlogloss']
# merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
STOPPING_METHOD = 'num_boost_round'
NUM_BOOST_ROUND = 100
EARLY_STOPPING_ROUNDS = 5
# endregion

# region params hierarchical clustering
REDUCTION = 0.2
PERCENTILES = [0.5, 1]
FIND_THRESH = False
SIMULATED_SAMPLES = 0
PLOT_RESULTS = True
CORES = mp.cpu_count()

DISTANCE_METRIC = 'matching'  # 'hamming'
# Distance is normalized so it ranges between [0, 1]
# (proportion of those vector elements between two n-vectors u and v which disagree.)
HC_METHOD = 'complete'
# 'complete' - Farthest Point Algorithm
TEMP_FOLDER = join(os.getcwd(), 'mlst_temp')
# endregion

# region params find recommended threshold
PERCENTILES_TO_CHECK = np.arange(.5, 20.5, 0.5)
# endregion

