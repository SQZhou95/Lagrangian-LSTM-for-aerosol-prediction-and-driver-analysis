"""
Train the final LightGBM model with the hyperparameters selected by
LightGBM_grid_search.py.

"""

import numpy as np
import scipy.io

from ml_utils import load_data, get_test_index
from train_validation_split import train_validation_split
from train_LightGBM import train_LightGBM
from stat_cal_mean import stat_cal_mean

# ---------------------------------------------------------------------------
# Load data for training
# ---------------------------------------------------------------------------
X, T, time_valid = load_data('data_for_traditional_ML.mat')

# T is the target variable, i.e. log10(NCCN).
# X is the input dataset. It can be the last time step on the trajectory or
# trajectory average (weighted or non-weighted).

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
index = {'Test': get_test_index(time_valid)}

para_split = {
    'block_size':   'daily',   # 'hourly','daily','weekly','monthly'
    'f_validation': 0.15,      # validation fraction
    'num_MCCV':     1,
}

rng = np.random.default_rng(0)
idx_train, idx_validation = train_validation_split(
    time_valid, index['Test'], para_split, rng=rng)

index['Train_validation'] = np.sort(np.union1d(idx_train[0], idx_validation[0]))

print('Samples: train+validation={}, test={}'.format(
    len(index['Train_validation']), len(index['Test'])))

# ---------------------------------------------------------------------------
# Set hyperparameters for LightGBM
#
# Values below are those reported in the manuscript SI. Replace them with the output of
# LightGBM_grid_search.py when using a different input dataset.
#   local conditions    : n_estimators=1000, num_leaves=127, lr=0.01, min_child_samples=50
#   trajectory averages : n_estimators=1000, num_leaves=31,  lr=0.01, min_child_samples=50
# ---------------------------------------------------------------------------
para = {
    'n_estimators':      1000,
    'num_leaves':        127,
    'learning_rate':     0.01,
    'min_child_samples': 50,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
}
num_try = 1   # number of repeated trainings

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
Mdl, Y, Stat = train_LightGBM(
    X, T, index, num_try=num_try,
    fit_key='Train_validation',
    eval_keys=('Train_validation', 'Test'), **para)

Y_mean, _, Stat_Y_mean, _ = stat_cal_mean(T[index['Test']], Y['Test'])

LightGBM = {
    'Y_Test':                 Y['Test'],
    'Y_mean_Test':            Y_mean,
    'RMSE_Train_validation':  Stat['RMSE']['Train_validation'],
    'R2_Train_validation':    Stat['R2']['Train_validation'],
    'Slope_Train_validation': Stat['Slope']['Train_validation'],
    'RMSE_Test':              Stat['RMSE']['Test'],
    'R2_Test':                Stat['R2']['Test'],
    'Slope_Test':             Stat['Slope']['Test'],
    'RMSE_mean_Test':         Stat_Y_mean['RMSE'],
    'R2_mean_Test':           Stat_Y_mean['R2'],
    'Slope_mean_Test':        Stat_Y_mean['Slope'],
}

print('\n=== Final result (ensemble mean of {} run(s)) ==='.format(num_try))
print('  RMSE_Test  = {:.4f}'.format(Stat_Y_mean['RMSE'][-1]))
print('  R2_Test    = {:.4f}'.format(Stat_Y_mean['R2'][-1]))
print('  Slope_Test = {:.4f}'.format(Stat_Y_mean['Slope'][-1]))

para_struct = {k: np.array([v]) for k, v in para.items()}
scipy.io.savemat('LightGBM_training.mat',
                 {'LightGBM': LightGBM, 'para': para_struct,
                  'index': {k: v + 1 for k, v in index.items()}})  # 1-based for MATLAB
print('\nSaved to LightGBM_training.mat')
