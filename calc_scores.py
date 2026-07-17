#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_scores.py
==============
Evaluation metrics for continuous emotion recognition.

Returns CCC, PCC, RMSE, and Spearman as a 4-element array. Spearman is
included in the return value because the dyadic pipeline accumulates and
averages all four scores across sessions and subjects.

Original author: Maximilian Schmitt <maximilian.schmitt@uni-passau.de>
Adapted for Python 3 and PEP-8 compliance.
"""

import numpy as np
from scipy.stats import spearmanr


def calc_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute CCC, PCC, RMSE, and Spearman between two sequences.

    Parameters
    ----------
    x:
        Ground-truth values (1-D NumPy array).
    y:
        Predicted values (1-D NumPy array, same length as *x*).

    Returns
    -------
    np.ndarray
        Array ``[CCC, PCC, RMSE, Spearman]``.

    Notes
    -----
    - **CCC** (Concordance Correlation Coefficient): measures both precision
      and accuracy; ranges from -1 to 1.
    - **PCC** (Pearson's Correlation Coefficient): measures linear correlation;
      ranges from -1 to 1.
    - **RMSE** (Root Mean Squared Error): measures absolute prediction error.
    - **Spearman**: non-parametric rank correlation; robust to outliers.

    Variance is computed with ``n-1`` denominator (unbiased / Matlab-consistent).
    """
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = np.nansum((x - x_mean) ** 2) / (len(x) - 1)
    y_var = np.nansum((y - y_mean) ** 2) / (len(y) - 1)

    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)

    ccc      = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)
    pcc      = covariance / (x_std * y_std)
    rmse     = np.sqrt(np.nanmean((x - y) ** 2))
    spearman, _ = spearmanr(x, y)

    return np.array([ccc, pcc, rmse, float(spearman)])
