#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelconvlstm.py
================
Two-stage model architecture for dyadic continuous emotion recognition.

Stage 1 — ``get_model_n1``:
    A 2-D CNN that processes individual subject features independently.
    Used to generate initial per-subject emotion predictions in Stage 1.

Stage 2 — ``get_model_n2``:
    A ConvLSTM that processes each subject's features concatenated with
    the partner's Stage 1 prediction. Captures long-term interpersonal
    affect dependencies across the dyad.

Architecture summary
--------------------
modelN1 (CNN):
    Conv2D(16, 1×21) → MaxPool2D(1×3, stride 4×1) → BatchNorm
    → Flatten → Dense(1024, ReLU) → Dropout(0.25) → BatchNorm
    → Dense(output)

modelN2 (ConvLSTM):
    ConvLSTM2D(16, 1×21) → MaxPool3D(1×1×3, stride 1×1×4) → BatchNorm
    → Flatten → Dense(1024, ReLU) → Dropout(0.25) → BatchNorm
    → Dense(output)

Loss: MSE | Optimizer: Adam
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: F401
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten,
    MaxPooling2D, MaxPooling3D,
)
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.models import Sequential


def get_model_n1(
    temporal_batch_size: int,
    n_features: int,
    class_num: int = 1,
) -> Sequential:
    """Build and compile the Stage 1 CNN model.

    Parameters
    ----------
    temporal_batch_size:
        Number of time frames per input window (height of the 2-D input).
    n_features:
        Number of features per frame (width of the 2-D input).
    class_num:
        Number of output dimensions (1 for single-dimension regression).

    Returns
    -------
    tensorflow.keras.models.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential(name="Stage1_CNN")

    model.add(Conv2D(
        16, kernel_size=(1, 21), padding="same", activation="relu",
        input_shape=(temporal_batch_size, n_features, 1),
        name="conv1",
    ))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(4, 1), name="pool1"))
    model.add(BatchNormalization(name="bn_conv1"))

    model.add(Flatten(name="flatten"))

    model.add(Dense(1024, activation="relu", name="fc1"))
    model.add(Dropout(0.25, name="drop1"))
    model.add(BatchNormalization(name="bn_fc1"))

    model.add(Dense(class_num, name="output"))

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def get_model_n2(
    temporal_batch_size: int,
    n_features: int,
    class_num: int = 1,
) -> Sequential:
    """Build and compile the Stage 2 ConvLSTM model.

    Takes input of shape ``(batch, 1, temporal_batch_size, n_features, 1)``
    where the second dimension is the sequence axis for the ConvLSTM.

    Parameters
    ----------
    temporal_batch_size:
        Number of time frames per input window.
    n_features:
        Number of features per frame. Should be original features + 1
        (for the partner's Stage 1 prediction concatenated as context).
    class_num:
        Number of output dimensions.

    Returns
    -------
    tensorflow.keras.models.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential(name="Stage2_ConvLSTM")

    model.add(ConvLSTM2D(
        16, kernel_size=(1, 21), padding="same",
        input_shape=(1, temporal_batch_size, n_features, 1),
        return_sequences=True, activation="tanh",
        name="convlstm1",
    ))
    model.add(MaxPooling3D(pool_size=(1, 1, 3), strides=(1, 1, 4), name="pool1"))
    model.add(BatchNormalization(name="bn_convlstm1"))

    model.add(Flatten(name="flatten"))

    model.add(Dense(1024, activation="relu", name="fc1"))
    model.add(Dropout(0.25, name="drop1"))
    model.add(BatchNormalization(name="bn_fc1"))

    model.add(Dense(class_num, name="output"))

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


# Backward-compatible aliases
getModelN1 = get_model_n1
getModelN2 = get_model_n2
