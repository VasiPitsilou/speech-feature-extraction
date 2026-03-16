# Speech Feature Extraction

A Jupyter notebook project for **speech emotion recognition** using audio feature extraction with [Librosa](https://librosa.org/) and [Praat (parselmouth)](https://parselmouth.readthedocs.io/).

## Overview

This project demonstrates how to extract acoustic features from speech audio files and use them to train machine learning classifiers that detect emotions. It also explores prosodic analysis using Praat.

The notebook (`librosa_praat.ipynb`) covers:

1. **Feature extraction with Librosa** – MFCC, chroma, and mel spectrogram features
2. **Emotion classification** – SVM and MLP classifiers trained on the [RAVDESS dataset](https://zenodo.org/record/1188976)
3. **Prosodic analysis with Praat** – intensity feature extraction and visualisation

## Dataset

The project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. Audio files are expected to be organised as:

```
Audio_Speech_Actors_01-24/
  Actor_01/
    03-01-01-01-01-01-01.wav
    ...
  Actor_02/
    ...
```

The filename encodes emotion via the **third** hyphen-separated segment. For example, `03-01-05-01-01-01-01.wav` has code `05` in position 3, which maps to **angry**:

| Code | Emotion   |
|------|-----------|
| 01   | neutral   |
| 02   | calm      |
| 03   | happy     |
| 04   | sad       |
| 05   | angry     |
| 06   | fearful   |
| 07   | disgust   |
| 08   | surprised |

## Features Extracted

For each audio file, the following features are computed with Librosa and concatenated into a single feature vector:

| Feature | Description |
|---------|-------------|
| MFCC (mean & std, 40 coefficients) | Timbre / spectral shape |
| Chroma (mean, 12 bins) | Pitch class energy |
| Mel spectrogram (mean) | Energy in mel-frequency bands |

## Models

| Model | Library | Notes |
|-------|---------|-------|
| SVM (RBF kernel) | scikit-learn | `C=10`, `gamma="scale"` |
| MLP | scikit-learn | Hidden layers `(128, 64)`, `max_iter=500` |

Features are standardised with `StandardScaler` before training. An 80/20 stratified train/test split is used.

## Praat Analysis

Using `parselmouth`, the notebook also:

- Extracts intensity features (mean, std, max) per audio file
- Plots and compares intensity contours between different emotions (e.g. angry vs. sad)

## Requirements

Install the dependencies before running the notebook:

```bash
pip install librosa numpy scikit-learn matplotlib parselmouth
```

## Usage

1. Download the RAVDESS dataset and place it in the project root as `Audio_Speech_Actors_01-24/`.
2. Open the notebook:

```bash
jupyter notebook librosa_praat.ipynb
```

3. Run all cells sequentially.

## Files

| File | Description |
|------|-------------|
| `librosa_praat.ipynb` | Main Jupyter notebook |
| `audio.wav` | Sample audio file used for the Librosa toy example |
| `LICENSE` | Apache 2.0 licence |

## License

This project is licensed under the [Apache License 2.0](LICENSE).
