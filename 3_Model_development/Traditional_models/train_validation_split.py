"""
Block-wise train/validation split.
"""

import numpy as np
from datetime import date


def _block_ids(time, block_size):
    """Reproduce the id_block_raw switch in train_validation_split.m."""
    if block_size == 'hourly':
        return np.round((time[:, 6] - time[0, 6]) * 24).astype(np.int64) + 1

    ymd = time[:, :3].astype(int)
    ordinals = np.fromiter(
        (date(int(y), int(m), int(d)).toordinal() for y, m, d in ymd),
        dtype=np.int64, count=len(ymd))

    if block_size == 'daily':
        return ordinals - ordinals[0] + 1
    if block_size == 'weekly':
        return np.floor((ordinals - ordinals[0] + 1.1) / 7).astype(np.int64) + 1
    if block_size == 'monthly':
        return 12 * (time[:, 0].astype(int) - int(time[0, 0])) + time[:, 1].astype(int)

    raise ValueError("Check para_split['block_size']! It must be one of: "
                     "hourly, daily, weekly, monthly.")


def train_validation_split(time, idx_test, para_split, rng=None):
    """Split the non-test samples into training and validation subsets.

    Parameters
    ----------
    time       : (N, 7) year, month, day, hour, minute, second, datenum
    idx_test   : 0-based indices of the test set
    para_split : dict with keys
                   block_size   : 'hourly' | 'daily' | 'weekly' | 'monthly'
                   f_validation : validation fraction (e.g. 0.15)
                   num_MCCV     : number of Monte-Carlo cross-validation
                                  realizations
    rng        : optional numpy Generator, for reproducibility

    Returns
    -------
    idx_train, idx_validation : lists of length num_MCCV holding 0-based
                                indices. For each realization the two subsets
                                are disjoint and together cover every non-test
                                sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = time.shape[0]
    idx_nontest = np.setdiff1d(np.arange(N), idx_test)

    id_block = _block_ids(time[idx_nontest], para_split['block_size'])
    id_block_unique = np.unique(id_block)

    # MATLAB round() is half-away-from-zero
    num_block_validation = int(np.floor(
        para_split['f_validation'] * len(id_block_unique) + 0.5))

    idx_train, idx_validation = [], []
    for _ in range(para_split['num_MCCV']):
        id_block_validation = rng.choice(id_block_unique,
                                         size=num_block_validation, replace=False)

        is_val = np.isin(id_block, id_block_validation)
        idx_validation.append(idx_nontest[is_val])
        idx_train.append(idx_nontest[~is_val])

    return idx_train, idx_validation
