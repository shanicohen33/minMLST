import multiprocessing as mp
import os
from os.path import join
import numpy as np

SEED = 10

# region params gene importance

OBJECTIVE = 'multi:softmax'
EVAL_METRIC = ['merror', 'mlogloss']
# merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).

# endregion

# region params hierarchical clustering

DISTANCE_METRIC = 'matching'  # 'hamming'
# Distance is normalized so it ranges between [0, 1]
# (proportion of those vector elements between two n-vectors u and v which disagree.)
HC_METHOD = 'complete'
# 'complete' - Farthest Point Algorithm
TEMP_FOLDER = join(os.getcwd(), 'mlst_temp')

# endregion

