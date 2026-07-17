#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/convlstmtest.py
===========================
Standalone demo script for a stacked ConvLSTM2D network on synthetically
generated moving-squares video data.

This script is **not part of the main dyadic CER pipeline**. It was used
during development to verify that the ConvLSTM2D layer works correctly
for spatiotemporal sequence prediction before integrating it into the
dyadic framework.

Reference
---------
    Original Keras ConvLSTM example:
    https://keras.io/examples/vision/conv_lstm/
"""

import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.models import Sequential

# Build a stacked ConvLSTM model for next-frame prediction
# Input shape: (n_frames, width, height, channels)
seq = Sequential(name="ConvLSTM_Demo")

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding="same", return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation="sigmoid",
               padding="same", data_format="channels_last"))

seq.compile(loss="binary_crossentropy", optimizer="adadelta")

print(seq.summary())
