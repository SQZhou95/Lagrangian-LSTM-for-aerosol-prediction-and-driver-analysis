"""
Grid search for the hyperparameters of LightGBM.

"""

import numpy as np
import scipy.io

from ml_utils import load_data, get_test_index
from train_validation_split import train_validation_split
from train_LightGBM import train_LightGBM

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

rng = np.random.default_rng(0)   # fixed seed so the search is reproducible
idx_train, idx_validation = train_validation_split(
    time_valid, index['Test'], para_split, rng=rng)

# As in train_RF.m, each candidate model is fitted on all non-test samples.
index['Train_validation'] = np.sort(np.union1d(idx_train[0], idx_validation[0]))

print('Samples: train+validation={}, test={}'.format(
    len(index['Train_validation']), len(index['Test'])))

# ---------------------------------------------------------------------------
# Set hyperparameters for the grid search
# ---------------------------------------------------------------------------
grid_search = {
    'n_estimators':      [100, 200, 500, 1000],
    'num_leaves':        [31, 63, 127],
    'learning_rate':     [0.01, 0.05, 0.1],
    'min_child_samples': [20, 50],
}
subsample = 0.8            # held fixed, not part of the search
colsample_bytree = 0.8     # held fixed, not part of the search

num_try = 1   # number of trainings per hyperparameter combination

# ---------------------------------------------------------------------------
# Grid search
#
# ---------------------------------------------------------------------------
results = []
best = None

for n_estimators in grid_search['n_estimators']:
    for num_leaves in grid_search['num_leaves']:
        for learning_rate in grid_search['learning_rate']:
            for min_child_samples in grid_search['min_child_samples']:

                Mdl, Y, Stat = train_LightGBM(
                    X, T, index,
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    min_child_samples=min_child_samples,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    num_try=num_try,
                    fit_key='Train_validation',
                    eval_keys=('Train_validation', 'Test'))

                row = {
                    'n_estimators':           n_estimators,
                    'num_leaves':             num_leaves,
                    'learning_rate':          learning_rate,
                    'min_child_samples':      min_child_samples,
                    'RMSE_Train_validation':  float(np.mean(Stat['RMSE']['Train_validation'])),
                    'R2_Train_validation':    float(np.mean(Stat['R2']['Train_validation'])),
                    'Slope_Train_validation': float(np.mean(Stat['Slope']['Train_validation'])),
                    'RMSE_Test':              float(np.mean(Stat['RMSE']['Test'])),
                    'R2_Test':                float(np.mean(Stat['R2']['Test'])),
                    'Slope_Test':             float(np.mean(Stat['Slope']['Test'])),
                }
                results.append(row)

                if best is None or row['RMSE_Test'] < best['RMSE_Test']:
                    best = row

                print('  n_estimators={}, num_leaves={}, learning_rate={}, '
                      'min_child_samples={} -> RMSE_Test={:.4f}, '
                      'R2_Test={:.4f}'.format(
                          n_estimators, num_leaves, learning_rate,
                          min_child_samples,
                          row['RMSE_Test'], row['R2_Test']))

# ---------------------------------------------------------------------------
# Report and save
# ---------------------------------------------------------------------------
print('\n=== Best hyperparameters ===')
for k, v in best.items():
    print('  {:22s} {}'.format(k, v))

fields = list(results[0].keys())
results_struct = {f: np.array([r[f] for r in results]) for f in fields}
best_struct = {f: np.array([best[f]]) for f in fields}

scipy.io.savemat('LightGBM_grid_search.mat',
                 {'results': results_struct,
                  'best': best_struct,
                  'grid_search': {k: np.array(v) for k, v in grid_search.items()}})
print('\nSaved to LightGBM_grid_search.mat')
