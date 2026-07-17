#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
write_predictions.py
====================
Write per-sequence model predictions to semicolon-delimited CSV files.

Original author: Maximilian Schmitt <maximilian.schmitt@uni-passau.de>
Adapted for Python 3 and PEP-8 compliance.
"""

import os

import numpy as np


def write_predictions(
    path_output: str,
    filename: str,
    predictions: np.ndarray,
    sr_labels: float = 0.1,
) -> None:
    """Write predictions for one sequence to a CSV file.

    Output format (semicolon-delimited)::

        'instance_name'; timestamp; pred_dim_0; pred_dim_1; ...

    Parameters
    ----------
    path_output:
        Directory in which to write the output file.
    filename:
        Name of the output CSV file (e.g. ``"April23_01.csv"``).
    predictions:
        Array of shape ``(n_dimensions, n_frames)`` containing the
        predicted values for each output dimension and time frame.
    sr_labels:
        Sampling rate of the predictions in seconds (default: 0.1 s = 10 Hz).
    """
    instance_name = os.path.splitext(filename)[0]
    output_path   = os.path.join(path_output, filename)

    with open(output_path, "w") as fh:
        n_frames = predictions.shape[1]
        for frame_idx in range(n_frames):
            timestamp = frame_idx * sr_labels
            row = f"'{instance_name}';{timestamp:.6f}"
            for dim in range(predictions.shape[0]):
                row += f";{predictions[dim, frame_idx]:.6f}"
            fh.write(row + "\n")
