"""
Performance metrics and cumulative ensemble statistics.

"""

import numpy as np
from scipy.stats import pearsonr


def metrics(T_ref, Y_ref):
    """RMSE, R2 (squared Pearson correlation) and regression slope.
    """
    T_ref = np.asarray(T_ref, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((T_ref - Y_ref) ** 2)))
    r2 = float(pearsonr(T_ref, Y_ref)[0] ** 2)
    slope = float(np.polyfit(T_ref, Y_ref, 1)[0])
    return rmse, r2, slope


def stat_cal_mean(T_eval, Y_eval_all, idx_rmse_sort=None):
    """Cumulative ensemble-mean predictions and their statistics.

    Parameters
    ----------
    T_eval        : (n_eval,) observed values
    Y_eval_all    : (n_eval, num_try) predictions from the individual runs
    idx_rmse_sort : optional ordering used for Y_best_mean.

    Returns
    -------
    Y_mean        : (n_eval, num_try); column i is the mean of runs 1..i+1
    Y_best_mean   : (n_eval, num_try) or None; column i is the mean of the
                    i+1 runs ranked first by idx_rmse_sort
    stat_Y_mean, stat_best_mean : dicts of RMSE / R2 / Slope arrays (num_try,)

    """
    num_try = Y_eval_all.shape[1]

    Y_mean = np.full_like(Y_eval_all, np.nan)
    stat_Y_mean = {k: np.full(num_try, np.nan) for k in ('RMSE', 'R2', 'Slope')}

    for i in range(num_try):
        y = np.mean(Y_eval_all[:, :i + 1], axis=1)
        Y_mean[:, i] = y
        stat_Y_mean['RMSE'][i], stat_Y_mean['R2'][i], stat_Y_mean['Slope'][i] = \
            metrics(T_eval, y)

    if idx_rmse_sort is None:
        return Y_mean, None, stat_Y_mean, None

    Y_best_mean = np.full_like(Y_eval_all, np.nan)
    stat_best_mean = {k: np.full(num_try, np.nan) for k in ('RMSE', 'R2', 'Slope')}

    for i in range(num_try):
        y = np.mean(Y_eval_all[:, idx_rmse_sort[:i + 1]], axis=1)
        Y_best_mean[:, i] = y
        stat_best_mean['RMSE'][i], stat_best_mean['R2'][i], stat_best_mean['Slope'][i] = \
            metrics(T_eval, y)

    return Y_mean, Y_best_mean, stat_Y_mean, stat_best_mean
