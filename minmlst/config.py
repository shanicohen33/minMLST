# todo- add joblib and multiprocessing to imports
import multiprocessing as mp

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
DISTANCE_METRIC = 'matching'  # 'hamming'
# Distance is normalized so it ranges between [0, 1]
# (proportion of those vector elements between two n-vectors u and v which disagree.)
HC_METHOD = 'complete'
# 'complete' - Farthest Point Algorithm
MAX_PERCENTILE = 20
# todo- set to 1000
SIMULATION_NUM_OF_SAMPLES = 10 #1000

