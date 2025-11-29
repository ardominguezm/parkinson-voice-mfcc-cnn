#!/usr/bin/env python3
"""
full_pipeline.py

Runs the full pipeline:
1. Load raw CSV signals
2. Extract MFCC spectral images
3. Build CNN model
4. Perform 10-fold cross-validation
5. Compute and save metrics
6. Visualize MFCC examples

This script is self-contained (does not import the numbered modules)
so that it can be executed independently.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras import layers, models


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_CONTROL_DIR = BASE_DIR / "data" / "raw" / "control"
RAW_PARKINSON_DIR = BASE_DIR / "data" / "raw" / "parkinson"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "analysis"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SR = 24000
N_MFCC = 40
HOP_LENGTH = 128


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_signals_from_dir(directory: Path, column_name: str = "Normalized Amplitude") -> List[np.ndarray]:
    """Load all CSV files from a directory as 1D numpy arrays."""
    signals = []
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for file in sorted(directory.glob("*.csv")):
        df = pd.read_csv(file)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in {file}")
        signals.append(df[column_name].values.astype(float))
    return signals


def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int, hop_length: int) -> np.ndarray:
    """Compute MFCC matrix (n_mfcc, T) for a given signal."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length
    )
    return mfcc


def pad_mfcc(mfcc: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad MFCC along the time axis to have exactly max_frames columns."""
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    return mfcc


def build_cnn_model(input_shape):
    """Define the CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    print("=== full_pipeline.py ===")
    print(f"Base directory: {BASE_DIR}")

    # 1) Load raw signals
    control_signals = load_signals_from_dir(RAW_CONTROL_DIR)
    parkinson_signals = load_signals_from_dir(RAW_PARKINSON_DIR)

    print(f"Loaded {len(control_signals)} control signals")
    print(f"Loaded {len(parkinson_signals)} Parkinson signals")

    all_signals = control_signals + parkinson_signals
    labels = np.array(
        [0] * len(control_signals) + [1] * len(parkinson_signals),
        dtype=int
    )

    # 2) Extract MFCCs
    print("Extracting MFCCs...")
    mfcc_list = [extract_mfcc(sig, SR, N_MFCC, HOP_LENGTH) for sig in all_signals]
    max_frames = max(m.shape[1] for m in mfcc_list)
    print(f"Maximum number of frames: {max_frames}")

    mfcc_padded = np.stack(
        [pad_mfcc(m, max_frames) for m in mfcc_list],
        axis=0
    )
    mfcc_padded = np.expand_dims(mfcc_padded, axis=-1)

    print(f"MFCC images shape: {mfcc_padded.shape}")
    print(f"Labels shape: {labels.shape}")

    # Save processed data
    np.save(PROCESSED_DIR / "mfcc_images.npy", mfcc_padded)
    np.save(PROCESSED_DIR / "labels.npy", labels)

    # 3) Cross-validation
    print("\n--- 10-fold cross-validation ---")
    X = mfcc_padded
    y = labels
    input_shape = X.shape[1:]

    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    metrics_dict = {
        "fold": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
    }

    fold_idx = 1
    for train_idx, val_idx in skf.split(X, y):
        print(f"\nFold {fold_idx}/10")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_cnn_model(input_shape)
        history = model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=1,
        )

        y_pred_prob = model.predict(X_val).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_prob)

        print(
            f"Fold {fold_idx} - "
            f"Accuracy: {acc:.4f} | F1: {f1:.4f} | "
            f"Precision: {prec:.4f} | Recall: {rec:.4f} | "
            f"ROC AUC: {roc_auc:.4f}"
        )

        metrics_dict["fold"].append(fold_idx)
        metrics_dict["accuracy"].append(acc)
        metrics_dict["f1"].append(f1)
        metrics_dict["precision"].append(prec)
        metrics_dict["recall"].append(rec)
        metrics_dict["roc_auc"].append(roc_auc)

        fold_idx += 1

    df_metrics = pd.DataFrame(metrics_dict)
    csv_path = ANALYSIS_DIR / "cv_metrics_full_pipeline.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved cross-validation metrics to: {csv_path}")

    # 4) Summary statistics
    print("\n--- Summary statistics ---")
    for col in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        vals = df_metrics[col].values
        mean_val = vals.mean()
        std_val = vals.std()
        cv = std_val / mean_val if mean_val != 0 else 0.0
        print(
            f"{col.capitalize()}: {mean_val:.4f} ± {std_val:.4f} "
            f"(CV = {cv:.4f})"
        )

    # 5) Visualization of MFCC examples
    print("\n--- Visualizing MFCC examples ---")
    idx_control_candidates = np.where(labels == 0)[0]
    idx_pd_candidates = np.where(labels == 1)[0]

    if len(idx_control_candidates) == 0 or len(idx_pd_candidates) == 0:
        print("Not enough samples per class for visualization.")
        return

    idx_control = int(idx_control_candidates[0])
    idx_pd = int(idx_pd_candidates[0])

    img_control = mfcc_padded[idx_control].squeeze()
    img_pd = mfcc_padded[idx_pd].squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(img_control, aspect="auto", origin="lower")
    axes[0].set_title("MFCC Spectrum - Healthy Control")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("MFCC Coefficients")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(img_pd, aspect="auto", origin="lower")
    axes[1].set_title("MFCC Spectrum - Parkinson's Disease")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("MFCC Coefficients")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = ANALYSIS_DIR / "mfcc_example_control_vs_pd_full_pipeline.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Saved MFCC example figure to: {fig_path}")


if __name__ == "__main__":
    main()
