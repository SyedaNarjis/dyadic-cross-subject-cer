#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
architectures/modelconv.py
==========================
Earlier draft of the two-stage model architecture. Functionally identical
to ``modelconvlstm.py`` but preserved here for reference.

Note: ``modelconvlstm.py`` is the version used in the main pipeline.
This file is kept for completeness only.
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: F401
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten,
    MaxPooling2D, MaxPooling3D,
)
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.models import Sequential


def get_model_n1(temporal_batch_size: int, n_features: int, class_num: int = 1) -> Sequential:
    """Stage 1 CNN — see modelconvlstm.get_model_n1 for full documentation."""
    model = Sequential(name="ModelConv_Stage1_CNN")
    model.add(Conv2D(16, kernel_size=(1, 21), padding="same", activation="relu",
                     input_shape=(temporal_batch_size, n_features, 1), name="conv1"))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(4, 1), name="pool1"))
    model.add(BatchNormalization(name="bn_conv1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(1024, activation="relu", name="fc1"))
    model.add(Dropout(0.25, name="drop1"))
    model.add(BatchNormalization(name="bn_fc1"))
    model.add(Dense(class_num, name="output"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def get_model_n2(temporal_batch_size: int, n_features: int, class_num: int = 1) -> Sequential:
    """Stage 2 ConvLSTM — see modelconvlstm.get_model_n2 for full documentation."""
    model = Sequential(name="ModelConv_Stage2_ConvLSTM")
    model.add(ConvLSTM2D(16, kernel_size=(1, 21), padding="same",
                         input_shape=(1, temporal_batch_size, n_features, 1),
                         return_sequences=True, activation="tanh", name="convlstm1"))
    model.add(MaxPooling3D(pool_size=(1, 1, 3), strides=(1, 1, 4), name="pool1"))
    model.add(BatchNormalization(name="bn_convlstm1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(1024, activation="relu", name="fc1"))
    model.add(Dropout(0.25, name="drop1"))
    model.add(BatchNormalization(name="bn_fc1"))
    model.add(Dense(class_num, name="output"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


getModelN1 = get_model_n1
getModelN2 = get_model_n2
