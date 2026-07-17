#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_convlstm.py
================
Dyadic continuous emotion recognition (CER) using a two-stage CNN + ConvLSTM
pipeline on the USC CreativeIT dataset with IS17 feature sets.

Pipeline
--------
Stage 1 — CNN (modelN1):
    Trained on all subjects jointly. Predicts emotion independently for
    Subject 1 (S1) and Subject 2 (S2) in each dyadic session.

Stage 2 — ConvLSTM (modelN2):
    Takes each subject's original features concatenated with the *partner's*
    Stage 1 prediction (cross-dyad affect context), capturing interpersonal
    emotion dynamics across the dyad.

Predictions are smoothed with a Savitzky-Golay filter and interpolated back
to the original frame rate before evaluation.

Evaluation
----------
Per-session and overall scores (CCC, PCC, RMSE, Spearman) are appended to
the results file specified by ``--results``.

Usage
-----
    python main_convlstm.py [OPTIONS]

    # Audio only
    python main_convlstm.py --no-motion

    # Audio + motion (default)
    python main_convlstm.py --audio --motion

    # Full options
    python main_convlstm.py \\
        --audio-path  /path/to/audio_features/ \\
        --motion-path /path/to/motion_features/ \\
        --label-path  /path/to/labels/ \\
        --results     results/results.txt \\
        --epochs      20 \\
        --batch-size  100

Dependencies
------------
    pip install tensorflow numpy scipy

References
----------
    CreativeIT dataset: https://sail.usc.edu/CreativeIT/ImprovDatabase.htm
    IS17 feature set:   http://www.compare.openaudio.eu/

Author  : (Syeda Narjis Fatima)
License : MIT
"""

import argparse
import fnmatch
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import interpolate
from tensorflow.keras.callbacks import EarlyStopping

from calc_scores import calc_scores
from load_features import d_2d_to_3d, load_all, win_mean
from modelconvlstm import get_model_n1, get_model_n2
from utils import savitzky_golay
from write_predictions import write_predictions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TEMPORAL_BATCH_SIZE   = 120
TEMPORAL_BATCH_STRIDE = 20
EPOCHS                = 20
BATCH_SIZE            = 100
RANDOM_SEED           = 7
SG_WINDOW             = 11    # Savitzky-Golay smoothing window (must be odd)
SG_ORDER              = 4     # Savitzky-Golay polynomial order
DELTA                 = 0     # Temporal delay between dyadic partners (seconds)

SESSIONS = ["April23*", "April30*"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-stage dyadic CER using CNN + ConvLSTM on CreativeIT IS17.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data paths ----------------------------------------------------------
    parser.add_argument("--audio-path",  default="features/audio/",
                        help="Directory containing audio feature CSV files.")
    parser.add_argument("--motion-path", default="features/motion/",
                        help="Directory containing motion feature CSV files.")
    parser.add_argument("--label-path",  default="features/labels/",
                        help="Directory containing label CSV files.")
    parser.add_argument("--pred-path",   default="test_predictions/",
                        help="Directory for writing test predictions.")
    parser.add_argument("--results",     default="results/results.txt",
                        help="Output file for session-wise and overall scores.")

    # -- Modality flags ------------------------------------------------------
    modality = parser.add_argument_group("modality flags")
    modality.add_argument("--audio",     dest="audio",    action="store_true",  default=True)
    modality.add_argument("--no-audio",  dest="audio",    action="store_false")
    modality.add_argument("--motion",    dest="motion",   action="store_true",  default=True)
    modality.add_argument("--no-motion", dest="motion",   action="store_false")
    modality.add_argument("--type",      dest="use_type", action="store_true",  default=True,
                          help="Include type feature column.")
    modality.add_argument("--no-type",   dest="use_type", action="store_false")
    modality.add_argument("--act",       dest="use_act",  action="store_true",  default=False,
                          help="Cross-subject activation column — disabled in all original experiments.")
    modality.add_argument("--no-act",    dest="use_act",  action="store_false")

    # -- Training hyperparameters --------------------------------------------
    train = parser.add_argument_group("training")
    train.add_argument("--epochs",      type=int,   default=EPOCHS)
    train.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    train.add_argument("--window-size", type=int,   default=TEMPORAL_BATCH_SIZE,
                       help="Temporal window length (frames).")
    train.add_argument("--hop",         type=int,   default=TEMPORAL_BATCH_STRIDE,
                       help="Stride between windows (frames).")
    train.add_argument("--delta",       type=float, default=DELTA,
                       help="Temporal delay between dyadic partners (seconds). "
                            "Positive = S1 leads, negative = S2 leads.")
    train.add_argument("--seed",        type=int,   default=RANDOM_SEED)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth_and_reshape(predictions: np.ndarray) -> np.ndarray:
    """Apply Savitzky-Golay smoothing and reshape to column vector."""
    flat = np.reshape(predictions, len(predictions))
    smoothed = savitzky_golay(flat, window_size=SG_WINDOW, order=SG_ORDER)
    return np.reshape(smoothed, (-1, 1))


def interpolate_predictions(pred: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate windowed predictions back to original frame rate."""
    x_old = np.arange(pred.shape[0])
    x_new = np.linspace(1, x_old[-1], target_len)
    func  = interpolate.splrep(x_old, pred, s=0)
    return interpolate.splev(x_new, func, der=0)[..., np.newaxis]


def apply_delay(pred1, pred2, data1, data2, labels1, labels2, delta, fps=60):
    """Roll predictions by delta seconds and trim rolled edges."""
    e       = pred1.shape[0]
    e_shift = int(delta * fps)

    pred1 = np.roll(pred1,      e_shift,    axis=0)
    pred2 = np.roll(pred2, -1 * e_shift,    axis=0)

    if delta < 0:
        r1 = range(e - e_shift, e)
        r2 = range(0, e_shift)
    else:
        r1 = range(0, e_shift)
        r2 = range(e - e_shift, e)

    pred2   = np.delete(pred1,   r1, axis=0)
    data1   = np.delete(data1,   r1, axis=0)
    labels1 = np.delete(labels1, r1, axis=0)

    pred1   = np.delete(pred1,   r2, axis=0)
    data2   = np.delete(data2,   r2, axis=0)
    labels2 = np.delete(labels2, r2, axis=0)

    return pred1, pred2, data1, data2, labels1, labels2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)

    if not args.audio and not args.motion:
        log.error("At least one modality (--audio or --motion) must be enabled.")
        return 1

    path_features = []
    if args.audio:
        path_features.append(args.audio_path)
    if args.motion:
        path_features.append(args.motion_path)

    Path(args.pred_path).mkdir(parents=True, exist_ok=True)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)

    # -- Log run metadata ----------------------------------------------------
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"\n{'=' * 60}\n"
        f"  Dyadic Cross-Subject CER — CreativeIT IS17\n"
        f"  Model      : CNN (Stage 1) + ConvLSTM (Stage 2)\n"
        f"  Modalities : audio={args.audio}, motion={args.motion}\n"
        f"  Pred dim   : Activation\n"
        f"  Run time   : {run_time}\n"
        f"{'=' * 60}\n"
    )
    log.info(header)
    with open(args.results, "a") as f:
        f.write(header)
        f.write(f"[Audio={args.audio}, Motion={args.motion}, "
                f"Type={args.use_type}, Act={args.use_act}]\n")

    # =========================================================================
    # Stage 1 — CNN: independent per-subject prediction
    # =========================================================================
    stage1_scores = np.zeros(4)   # CCC, PCC, RMSE, Spearman
    session_data   = []           # collects cross-dyad features for Stage 2
    session_labels = []

    for sess in SESSIONS:
        log.info("Stage 1 — test session: %s", sess)

        train_sessions = [s for s in SESSIONS if s != sess]
        files_test  = sorted(fnmatch.filter(os.listdir(path_features[0]), sess))
        files_train = [
            f
            for t_sess in train_sessions
            for f in fnmatch.filter(os.listdir(path_features[0]), t_sess)
        ]

        # Load features and labels
        train_data    = load_all(files_train, path_features, args.use_type, args.use_act)
        test_sep      = load_all(files_test,  path_features, args.use_type, args.use_act,
                                 separate=True)
        train_labels  = load_all(files_train, [args.label_path], True, True)
        test_labels_sep = load_all(files_test, [args.label_path], True, True,
                                   separate=True)

        # Reshape into temporal windows
        train_feats  = d_2d_to_3d(train_data, args.window_size, args.hop)[..., np.newaxis]
        train_labels_w = win_mean(train_labels, args.window_size, args.hop)
        n_feats = train_feats.shape[2]

        # Build and train Stage 1 model
        model_n1 = get_model_n1(args.window_size, n_feats, class_num=1)
        log.info(model_n1.summary())
        model_n1.fit(
            train_feats, train_labels_w,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            validation_split=0.25,
            shuffle=True,
        )

        # -- Per-dyad pair evaluation ----------------------------------------
        sess_scores = np.zeros(4)
        data_sess   = np.zeros([1, n_feats + 1])
        labels_sess = np.zeros([1, 1])

        for f in range(0, len(files_test), 2):
            test1    = test_sep[f]
            test2    = test_sep[f + 1]
            labels1  = test_labels_sep[f]
            labels2  = test_labels_sep[f + 1]

            # Windowed features for prediction
            feats1 = d_2d_to_3d(test1, args.window_size, args.hop)[..., np.newaxis]
            feats2 = d_2d_to_3d(test2, args.window_size, args.hop)[..., np.newaxis]
            win_labels1 = win_mean(labels1, args.window_size, args.hop)
            win_labels2 = win_mean(labels2, args.window_size, args.hop)

            # Stage 1 predictions + smoothing
            pred1 = smooth_and_reshape(model_n1.predict(feats1))
            pred2 = smooth_and_reshape(model_n1.predict(feats2))

            sess_scores += np.abs(calc_scores(win_labels1, pred1))
            sess_scores += np.abs(calc_scores(win_labels2, pred2))

            # Interpolate back to original frame rate
            pred_intp1 = interpolate_predictions(pred1, test1.shape[0])
            pred_intp2 = interpolate_predictions(pred2, test2.shape[0])

            # Apply temporal delay between partners
            pred_intp1, pred_intp2, test1, test2, labels1, labels2 = apply_delay(
                pred_intp1, pred_intp2, test1, test2, labels1, labels2, args.delta
            )

            # Concatenate: S1 features + S2 cross-dyad prediction → S2 input for Stage 2
            data_temp   = np.concatenate([test1, pred_intp2], axis=1)
            data_sess   = np.concatenate([data_sess, data_temp], axis=0)
            labels_sess = np.concatenate([labels_sess, labels1], axis=0)

            data_temp   = np.concatenate([test2, pred_intp1], axis=1)
            data_sess   = np.concatenate([data_sess, data_temp], axis=0)
            labels_sess = np.concatenate([labels_sess, labels2], axis=0)

        sess_scores /= len(files_test)
        stage1_scores += np.abs(sess_scores)

        log.info("Stage 1 session %s — CCC: %.4f  PCC: %.4f  RMSE: %.4f  Spearman: %.4f",
                 sess, *sess_scores)

        with open(args.results, "a") as f:
            f.write(f"Stage 1 — session: {sess}\n")
            f.write(f"CCC / PCC / RMSE / Spearman: {sess_scores}\n")

        session_data.append(data_sess)
        session_labels.append(labels_sess)

    stage1_mean = stage1_scores / len(SESSIONS)
    log.info("Stage 1 overall — CCC: %.4f  PCC: %.4f  RMSE: %.4f  Spearman: %.4f",
             *stage1_mean)
    with open(args.results, "a") as f:
        f.write(f"\nStage 1 overall average:\n{stage1_mean}\n")

    # =========================================================================
    # Stage 2 — ConvLSTM: cross-dyad affect context
    # =========================================================================
    stage2_scores = np.zeros(4)

    for sess_idx in range(len(session_data)):
        log.info("Stage 2 — test session index: %d", sess_idx)

        # Build train/test split from accumulated session data
        train_data   = np.zeros([1, n_feats + 1])
        train_labels = np.zeros([1, 1])

        for i in range(len(session_data)):
            if i != sess_idx:
                train_data   = np.concatenate([train_data,   session_data[i]],   axis=0)
                train_labels = np.concatenate([train_labels, session_labels[i]], axis=0)

        test_data   = session_data[sess_idx]
        test_labels = session_labels[sess_idx]

        # Reshape into temporal windows + add sequence dimension for ConvLSTM
        train_feats   = d_2d_to_3d(train_data, args.window_size, args.hop)[..., np.newaxis]
        train_feats   = train_feats[:, np.newaxis, ...]
        train_labels_w = win_mean(train_labels, args.window_size, args.hop)

        test_feats    = d_2d_to_3d(test_data, args.window_size, args.hop)[..., np.newaxis]
        test_feats    = test_feats[:, np.newaxis, ...]
        test_labels_w = win_mean(test_labels, args.window_size, args.hop)

        # Build and train Stage 2 model
        model_n2 = get_model_n2(args.window_size, n_feats + 1, class_num=1)
        model_n2.fit(
            train_feats, train_labels_w,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            validation_split=0.25,
            shuffle=True,
        )

        # Stage 2 predictions + smoothing
        pred = smooth_and_reshape(model_n2.predict(test_feats))
        score = np.abs(calc_scores(test_labels_w, pred))
        stage2_scores += score

        log.info("Stage 2 session %d — CCC: %.4f  PCC: %.4f  RMSE: %.4f  Spearman: %.4f",
                 sess_idx, *score)
        with open(args.results, "a") as f:
            f.write(f"Stage 2 — session {SESSIONS[sess_idx]}:\n")
            f.write(f"CCC / PCC / RMSE / Spearman: {score}\n")

    stage2_mean = stage2_scores / len(session_data)
    log.info("Stage 2 overall — CCC: %.4f  PCC: %.4f  RMSE: %.4f  Spearman: %.4f",
             *stage2_mean)
    with open(args.results, "a") as f:
        f.write(f"\nStage 2 overall average:\n{stage2_mean}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
