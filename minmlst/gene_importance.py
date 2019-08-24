import numpy as np
import xgboost as xgb
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import minmlst.config as c


def calc_shap_values(bst, X, y):
    print("  mean_shap_values")
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X=X, y=y)
    mean_shap_values = np.sum(np.mean(np.abs(shap_values), axis=0), axis=0)
    gene_importance = pd.DataFrame({'gene': list(X.columns.values),
                                    'importance_by_shap': list(mean_shap_values)})
    return gene_importance


def calc_importance(bst, measure):
    print(f"  {measure}")
    scores = bst.get_score(importance_type=measure)
    gene_importance = pd.DataFrame({'gene': list(scores.keys()),
                                    'importance_by_' + measure: list(scores.values())})
    return gene_importance


# todo- filter singletones
def get_gene_importance(X, y, measures):
    # encode y labels
    lbl = LabelEncoder()
    lbl.fit(y)
    y_enc = lbl.transform(y)
    n_classes = len(np.unique(y_enc))

    # split into train and test (stratified)
    skf = StratifiedKFold(n_splits=2, random_state=c.SEED, shuffle=False)
    for train_index, test_index in skf.split(X=X, y=y_enc):
        if len(train_index) < len(test_index):
            tmp = train_index
            train_index = test_index
            test_index = tmp

        X_train = X.iloc[list(train_index), :]
        y_train = y_enc[train_index]
        X_test = X.iloc[list(test_index), :]
        y_test = y_enc[test_index]
        break
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # train an xgboost model
    params = {
        'seed': c.SEED,
        'objective': c.OBJECTIVE,  #todo- to be set by the user
        'num_class': n_classes,
        'max_depth': c.MAX_DEPTH,  #todo- to be set by the user
        'eta': c.LEARNING_RATE,  #todo- to be set by the user
        'verbose_eval': True,
        "silent": 1,
        'eval_metric': c.EVAL_METRIC  #todo- to be set by the user
    }
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    print("training a model")
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=c.NUM_BOOST_ROUND, #todo- to be set by the user
        # early_stopping_rounds=c.EARLY_STOPPING_ROUNDS,
        evals=watchlist,
        verbose_eval=10,
        evals_result=evals_result
    )

    print(f"calculating gene importance by:")
    scores_df = pd.DataFrame()
    for m in measures:
        if m == "shap":
            scores = calc_shap_values(bst=bst, X=X, y=y_enc)
        else:
            scores = calc_importance(bst=bst, measure=m)
        if scores_df.empty:
            scores_df = scores
            scores_df = scores_df.sort_values(by=scores_df.columns.values[1], ascending=True).reset_index(drop=True)
            # todo- sort again with ascending=False before clustering
        else:
            scores_df = pd.merge(scores_df, scores, how='outer', on='gene')

    return scores_df
