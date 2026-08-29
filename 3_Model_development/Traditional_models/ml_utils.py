"""
Shared helpers for the XGBoost / LightGBM scripts.

MATLAB has `load` and `disp` built in; this module provides the Python
equivalents used by the *_training.py and *_grid_search.py scripts.

Requirements:
    pip install numpy scipy xgboost lightgbm
    pip install h5py          # only if data_for_traditional_ML.mat is v7.3
"""

from datetime import datetime

import numpy as np
import scipy.io


def load_data(mat_path='data_for_traditional_ML.mat'):
    """Load the pre-processed training data.

    Parameters
    ----------
    mat_path : path to data_for_traditional_ML.mat

    Returns
    -------
    X          : (N, n_features) input matrix. Either the last timestep on the
                 trajectory or a trajectory average (weighted or non-weighted).
    T          : (N,) target vector, i.e. log10(NCCN).
    time_valid : (N, 7) year, month, day, hour, minute, second, datenum.
    """
    try:
        raw = scipy.io.loadmat(mat_path, squeeze_me=True)
        X = np.asarray(raw['X'], dtype=np.float64)
        T = np.asarray(raw['T'], dtype=np.float64).ravel()
        time_valid = np.asarray(raw['time_valid'], dtype=np.float64)
    except NotImplementedError:
        # v7.3 / HDF5: dimensions come back transposed
        import h5py
        with h5py.File(mat_path, 'r') as f:
            X = np.asarray(f['X'], dtype=np.float64)
            T = np.asarray(f['T'], dtype=np.float64).ravel()
            time_valid = np.asarray(f['time_valid'], dtype=np.float64)
        if X.shape[0] != T.shape[0]:
            X = X.T
        if time_valid.shape[0] != T.shape[0]:
            time_valid = time_valid.T

    if X.ndim == 1:
        X = X[:, None]
    assert X.shape[0] == T.shape[0] == time_valid.shape[0], \
        'X, T and time_valid must have the same number of rows'
    return X, T, time_valid


def get_test_index(time_valid):
    """
    Six months of 2022 covering all seasons are held out as the independent
    test set. Returns 0-based indices.
    """
    year = time_valid[:, 0].astype(int)
    month = time_valid[:, 1].astype(int)
    return np.where((year == 2022) & np.isin(month, [1, 3, 5, 7, 9, 11]))[0]


def print_result(label, i, stat, eval_key='Test'):
    """Progress line, equivalent to the disp() calls in train_RF.m."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{}  {} | run {} | RMSE_{}={:.4f}  R2_{}={:.4f}  Slope_{}={:.4f}'.format(
        ts, label, i + 1,
        eval_key, stat['RMSE'][eval_key][i],
        eval_key, stat['R2'][eval_key][i],
        eval_key, stat['Slope'][eval_key][i]), flush=True)
