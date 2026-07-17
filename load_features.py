#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_features.py
================
Utilities for loading and reshaping CSV feature files into NumPy arrays
suitable for temporal CNN training.

Original author: Maximilian Schmitt <maximilian.schmitt@uni-passau.de>
Adapted for Python 3 and PEP-8 compliance.
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------

def win_mean(x: np.ndarray, agg_num: int, hop: int) -> np.ndarray:
    """Compute the per-window mean of a 2-D array.

    Converts a ``(T, F)`` array into ``(N_windows, F)`` by taking the
    column-wise mean of each non-overlapping / overlapping window.

    Parameters
    ----------
    x:
        Input array of shape ``(T, F)``.
    agg_num:
        Window length in frames.
    hop:
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Shape ``(N_windows, F)``.
    """
    len_x, n_in = x.shape
    if len_x < agg_num:
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    len_x = len(x)
    windows = []
    i = 0
    while i + agg_num <= len_x:
        windows.append(np.mean(x[i: i + agg_num], axis=0))
        i += hop

    return np.array(windows)


def d_2d_to_3d(x: np.ndarray, agg_num: int, hop: int) -> np.ndarray:
    """Slice a ``(T, F)`` array into overlapping 3-D windows.

    Parameters
    ----------
    x:
        Input array of shape ``(T, F)``.
    agg_num:
        Window length in frames.
    hop:
        Stride between consecutive windows.

    Returns
    -------
    np.ndarray
        Shape ``(N_windows, agg_num, F)``.
    """
    len_x, n_in = x.shape
    if len_x < agg_num:
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    len_x = len(x)
    windows = []
    i = 0
    while i + agg_num <= len_x:
        windows.append(x[i: i + agg_num])
        i += hop

    return np.array(windows)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_features(filenames: list, path: str, num_lines: int) -> np.ndarray:
    """Load semicolon-delimited feature CSV files into a 2-D array.

    .. deprecated::
        This function is no longer called internally. Use ``_load_features_shift``
        via ``load_all`` instead. Kept for backwards compatibility only.

    Each CSV row has the format::

        instance_name ; timestamp ; feat_1 ; feat_2 ; ...

    The first two fields (instance name and timestamp) are skipped.

    Parameters
    ----------
    filenames:
        List of CSV filenames (basenames only) to load in order.
    path:
        Directory containing the CSV files.
    num_lines:
        Pre-computed total number of rows across all files.

    Returns
    -------
    np.ndarray
        Shape ``(num_lines, n_features)``.
    """
    n_cols   = _get_num_columns(os.path.join(path, filenames[0])) - 2
    features = np.empty((num_lines, n_cols), dtype=float)
    c = 0
    for fname in filenames:
        with open(os.path.join(path, fname), "r") as fh:
            for line in fh:
                pos = line.find(";", line.find(";") + 1)  # skip 2 header fields
                features[c] = np.fromstring(line[pos + 1:], dtype=float, sep=";")
                c += 1
    return features


def load_all(
    filenames: list,
    paths: list,
    b_type: bool,
    b_act: bool,
    shift: list = None,
    separate: bool = False,
) -> np.ndarray:
    """Load and concatenate features from multiple directories.

    Loads CSV files listed in *filenames* from each directory in *paths* and
    concatenates them along the feature axis. Optionally removes the type and
    activation cross-subject columns from the end of the feature vector.

    Parameters
    ----------
    filenames:
        Basenames of the CSV files to load.
    paths:
        List of directories; each contributes one block of features.
    b_type:
        Whether to keep the second-to-last feature column (type flag).
    b_act:
        Whether to keep the last feature column (activation flag).
    shift:
        Per-path temporal shift in frames (list of ints). Defaults to all zeros.
    separate:
        If ``True``, returns an object array where each element holds features
        for one sequence separately (used for sequence-level evaluation).

    Returns
    -------
    np.ndarray
        Concatenated feature array.
    """
    if shift is None:
        shift = np.zeros(len(paths), dtype=int)

    if separate:
        F = np.empty(len(filenames), dtype=object)
        for seq_idx, fname in enumerate(filenames):
            n_lines = _get_num_lines(os.path.join(paths[0], fname))
            F[seq_idx] = _load_features_shift([fname], paths[0], [n_lines], shift[0])
            if not b_type:
                F[seq_idx] = np.delete(F[seq_idx], F[seq_idx].shape[1] - 2, axis=1)
            if not b_act:
                F[seq_idx] = np.delete(F[seq_idx], F[seq_idx].shape[1] - 1, axis=1)
            for p_idx in range(1, len(paths)):
                Fn = _load_features_shift([fname], paths[p_idx], [n_lines], shift[p_idx])
                F[seq_idx] = np.concatenate((F[seq_idx], Fn), axis=1)
        return F

    # Concatenate all sequences into one array
    num_lines = _get_num_lines_array(filenames, paths[0])
    F = _load_features_shift(filenames, paths[0], num_lines, shift[0])
    for p_idx in range(1, len(paths)):
        Fn = _load_features_shift(filenames, paths[p_idx], num_lines, shift[p_idx])
        F = np.concatenate((F, Fn), axis=1)

    if not b_type:
        F = np.delete(F, F.shape[1] - 2, axis=1)
    if not b_act:
        F = np.delete(F, F.shape[1] - 1, axis=1)

    return F


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_features_shift(
    filenames: list, path: str, num_lines: np.ndarray, shift: int
) -> np.ndarray:
    """Load features with optional temporal forward-shift (padding at start)."""
    total_lines = int(np.sum(num_lines))
    n_cols      = _get_num_columns(os.path.join(path, filenames[0])) - 2
    features    = np.empty((total_lines, n_cols), dtype=float)

    f_idx = 0
    c     = 0
    for fname in filenames:
        c_start = c
        with open(os.path.join(path, fname), "r") as fh:
            for line in fh:
                pos = line.find(";", line.find(";") + 1)
                fv  = np.fromstring(line[pos + 1:], dtype=float, sep=";")
                if c == c_start and shift > 0:
                    for _ in range(shift):
                        features[c] = fv
                        c += 1
                elif (c - c_start) == num_lines[f_idx]:
                    break
                else:
                    features[c] = fv
                    c += 1
        f_idx += 1

    return features


def _get_num_lines_array(filenames: list, path: str) -> np.ndarray:
    counts = np.zeros(len(filenames), dtype=int)
    for i, fname in enumerate(filenames):
        counts[i] = _get_num_lines(os.path.join(path, fname))
    return counts


def _get_num_lines(filepath: str) -> int:
    with open(filepath, "r") as fh:
        return sum(1 for _ in fh)


def _get_num_columns(filepath: str, delim: str = ";") -> int:
    with open(filepath, "r") as fh:
        line = fh.readline()
    return line.count(delim) + 1
