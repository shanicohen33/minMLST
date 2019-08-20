import shap
import xgboost as xgb
import pandas as pd
from xgboost import XGBClassifier
import time
from os.path import join
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np


def test_func_double(x):
    print('XGBClassifier start')
    model = XGBClassifier(random_state=7, nthread=-1, importance_type='gain')
    print('XGBClassifier end')

    return 2*x


