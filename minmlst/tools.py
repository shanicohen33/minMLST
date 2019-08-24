"""
Machine learning module for finding a minimal MLST scheme for bacterial strain typing
=====================================================================================

TBD
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

import shap
import xgboost as xgb
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import time
from os.path import join
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np

__version__ = '0.0.1'

from minmlst.gene_importance import *


def test_func_double(x):
    print('XGBClassifier start')
    model = XGBClassifier(random_state=7, nthread=-1, importance_type='gain')
    print('XGBClassifier end')

    print('xgb.DMatrix start')
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([1, 1, 0])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    print('xgb.DMatrix end')

    return 2*x


def check(x):
    return check_2(x)


## fill missing values
## separate to X and ST
## get gene importance:

# min cluster size = 2




