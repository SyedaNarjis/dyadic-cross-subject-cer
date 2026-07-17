# Dyadic Cross-Subject CER

A two-stage CNN + ConvLSTM framework for continuous emotion recognition (CER)
in dyadic interactions, using the [USC CreativeIT dataset](https://sail.usc.edu/CreativeIT/ImprovDatabase.htm)
with IS17 feature sets.

Unlike single-subject approaches, this framework models **interpersonal emotion
dynamics** — each subject's emotion prediction is informed by their partner's
affective state, captured via a cross-dyad context injection between Stage 1
and Stage 2.

---

## Pipeline

```
Stage 1 — CNN (modelN1)
    Audio + motion features (S1) → CNN → S1 emotion prediction
    Audio + motion features (S2) → CNN → S2 emotion prediction
                    ↓ Savitzky-Golay smoothing + interpolation
                    ↓ Cross-dyad injection
                    S1 prediction → appended to S2 features
                    S2 prediction → appended to S1 features

Stage 2 — ConvLSTM (modelN2)
    S1 features + S2 context → ConvLSTM → refined S1 prediction
    S2 features + S1 context → ConvLSTM → refined S2 prediction
```

Evaluation: **CCC, PCC, RMSE, Spearman** per session and overall average.

---

## Requirements

| Package    | Tested version |
|------------|---------------|
| Python     | ≥ 3.9         |
| TensorFlow | ≥ 2.10        |
| NumPy      | ≥ 1.24        |
| SciPy      | ≥ 1.10        |

```bash
pip install tensorflow numpy scipy
```

---

## Project Structure

```
dyadic-cross-subject-cer/
├── main_convlstm.py        # Entry point — two-stage training & evaluation
├── modelconvlstm.py        # Stage 1 CNN + Stage 2 ConvLSTM architectures
├── load_features.py        # Feature loading and temporal windowing utilities
├── calc_scores.py          # CCC / PCC / RMSE / Spearman metrics
├── write_predictions.py    # Save test predictions to CSV
├── utils.py                # Savitzky-Golay smoothing filter
├── requirements.txt
├── README.md
├── results/                # Output scores written here (not committed)
│   └── .gitkeep
├── architectures/
│   └── modelconv.py        # Earlier draft architecture (reference only)
└── experiments/
    └── convlstmtest.py     # Standalone ConvLSTM demo (not part of pipeline)
```

---

## Usage

```bash
# Audio + motion (default)
python main_convlstm.py \
    --audio-path  /path/to/audio_features/ \
    --motion-path /path/to/motion_features/ \
    --label-path  /path/to/labels/

# Audio only
python main_convlstm.py --no-motion --audio-path /path/to/audio_features/ \
    --label-path /path/to/labels/
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio-path` | `features/audio/` | Audio feature CSV directory |
| `--motion-path` | `features/motion/` | Motion feature CSV directory |
| `--label-path` | `features/labels/` | Label CSV directory |
| `--results` | `results/results.txt` | Output file for scores |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `100` | Mini-batch size |
| `--window-size` | `120` | Temporal window length (frames) |
| `--hop` | `20` | Stride between windows (frames) |
| `--delta` | `0.0` | Temporal delay between partners (seconds) |
| `--audio / --no-audio` | on | Use audio modality |
| `--motion / --no-motion` | on | Use motion modality |
| `--type / --no-type` | on | Include type feature column |
| `--act / --no-act` | off | Cross-subject activation column — disabled in all original experiments |

---

## Related

- [continuous-emotion-recognition-cnn](https://github.com/SyedaNarjis/continuous-emotion-recognition-cnn) — single-subject CER baseline this work extends

---

## Data

Feature files follow the semicolon-delimited IS17 CSV format produced by
[openSMILE](https://www.audeering.com/research/opensmile/). Data is **not
included** in this repository. See the
[CreativeIT dataset page](https://sail.usc.edu/CreativeIT/ImprovDatabase.htm)
for access or email author for working dataset. 

---

## License

MIT
