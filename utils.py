#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
Utility functions for signal smoothing used in post-processing emotion
predictions.

Functions
---------
savitzky_golay:
    Smooth (and optionally differentiate) a signal using a Savitzky-Golay
    least-squares polynomial filter. Applied to Stage 1 and Stage 2
    predictions to reduce high-frequency noise before evaluation.
"""

from math import factorial

import numpy as np


def savitzky_golay(
    y: np.ndarray,
    window_size: int,
    order: int,
    deriv: int = 0,
    rate: float = 1.0,
) -> np.ndarray:
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high-frequency noise from data while
    preserving the original shape and features of the signal better than
    moving average techniques.

    Parameters
    ----------
    y:
        Values of the time history of the signal, shape ``(N,)``.
    window_size:
        Length of the filter window. Must be a positive odd integer.
    order:
        Order of the polynomial used for fitting. Must be less than
        ``window_size - 1``.
    deriv:
        Order of the derivative to compute. Default 0 = smoothing only.
    rate:
        Scaling factor for the derivative. Default 1.

    Returns
    -------
    np.ndarray
        Smoothed signal (or its n-th derivative), shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``window_size`` or ``order`` are not integers.
    TypeError
        If ``window_size`` is not a positive odd number, or if
        ``window_size`` is too small for the polynomial order.

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical Chemistry,
       1964, 36(8), pp 1627–1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing,
       W.H. Press et al., Cambridge University Press, ISBN 9780521880688.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp(-t**2) + np.random.normal(0, 0.05, t.shape)
    >>> y_smooth = savitzky_golay(y, window_size=31, order=4)
    """
    try:
        window_size = int(np.abs(window_size))
        order       = int(np.abs(order))
    except ValueError:
        raise ValueError("window_size and order must be of type int.")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be a positive odd number.")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomial order.")

    order_range  = range(order + 1)
    half_window  = (window_size - 1) // 2

    # Precompute Savitzky-Golay coefficients
    b = np.mat([
        [k ** i for i in order_range]
        for k in range(-half_window, half_window + 1)
    ])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

    # Pad signal at the extremes with reflected values
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals  = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y_padded  = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y_padded, mode="valid")
