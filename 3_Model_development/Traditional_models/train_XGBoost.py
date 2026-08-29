"""
Core XGBoost training function.
"""

import numpy as np
import xgboost as xgb

from ml_utils import print_result
from stat_cal_mean import metrics


def train_XGBoost(X, T, index,
                  n_estimators, max_depth, learning_rate,
                  subsample=0.8, colsample_bytree=0.8,
                  num_try=1,
                  fit_key='Train_validation',
                  eval_keys=('Train_validation', 'Test'),
                  verbose=True):
    """Fit XGBoost regressors and evaluate them on the requested subsets.

    Parameters
    ----------
    X                : (N, n_features) input matrix
    T                : (N,) target vector, log10(NCCN)
    index            : dict of 0-based index arrays; must contain fit_key and
                       every entry of eval_keys
    n_estimators     : number of boosting rounds     (cf. NumLearningCycles)
    max_depth        : maximum tree depth
    learning_rate    : step size shrinkage           (cf. LearnRate)
    subsample        : row subsampling ratio
    colsample_bytree : column subsampling ratio
    num_try          : number of repeated trainings; run i uses random_state=i.
                       Repeats differ only through the stochastic row/column
                       subsampling, so num_try>1 gives a genuine (if narrow)
                       ensemble.
    fit_key          : subset used for fitting. 'Train_validation' reproduces
                       the final-model protocol of train_RF.m; use 'Train' when
                       selecting hyperparameters against a held-out validation
                       set.
    eval_keys        : subsets on which predictions and statistics are returned

    Returns
    -------
    Mdl  : list of the num_try fitted models
    Y    : dict, Y[key] is (n_key, num_try) of predictions
    Stat : dict, Stat[metric][key] is (num_try,) for metric in RMSE/R2/Slope
    """
    for key in (fit_key,) + tuple(eval_keys):
        if key not in index:
            raise KeyError("index is missing the key '{}'".format(key))

    X_fit = X[index[fit_key], :]
    T_fit = T[index[fit_key]]

    Y = {k: np.full((len(index[k]), num_try), np.nan) for k in eval_keys}
    Stat = {m: {k: np.full(num_try, np.nan) for k in eval_keys}
            for m in ('RMSE', 'R2', 'Slope')}

    Mdl = []
    label = ('n_estimators={}, max_depth={}, learning_rate={}, '
             'subsample={}, colsample_bytree={}').format(
                 n_estimators, max_depth, learning_rate, subsample, colsample_bytree)

    for i in range(num_try):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method='hist',
            n_jobs=-1,
            random_state=i,
            verbosity=0,
        )
        model.fit(X_fit, T_fit)
        Mdl.append(model)

        for key in eval_keys:
            y = model.predict(X[index[key], :])
            Y[key][:, i] = y
            rmse, r2, slope = metrics(T[index[key]], y)
            Stat['RMSE'][key][i] = rmse
            Stat['R2'][key][i] = r2
            Stat['Slope'][key][i] = slope

        if verbose:
            print_result(label, i, Stat, eval_key=eval_keys[-1])

    return Mdl, Y, Stat
