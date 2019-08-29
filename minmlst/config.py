SEED = 10

OBJECTIVE = 'multi:softmax'
MAX_DEPTH = 6
LEARNING_RATE = 0.3
EVAL_METRIC = ['merror', 'mlogloss']
# merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
STOPPING_METHOD = 'num_boost_round'
NUM_BOOST_ROUND = 100
EARLY_STOPPING_ROUNDS = 5

