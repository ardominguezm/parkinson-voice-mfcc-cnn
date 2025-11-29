#!/usr/bin/env python3
"""
01_extract_mfcc.py

Extracts MFCC spectral images from raw normalized voice signals
and saves them as NumPy arrays in data/processed/.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import librosa


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_CONTROL_DIR = BASE_DIR / "data" / "raw" / "control"
RAW_PARKINSON_DIR = BASE_DIR / "data" / "raw" / "parkinson"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SR = 24000       # Sampling rate (Hz)
N_MFCC = 40      # Number of MFCC coefficients
HOP_LENGTH = 128 # Frame hop length


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_signals_from_dir(directory: Path, column_name: str = "Normalized Amplitude") -> List[np.ndarray]:
    """Load all CSV files in a directory as 1D numpy arrays."""
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
    """Compute MFCC matrix of shape (n_mfcc, T) for a 1D signal."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length
    )
    return mfcc


def pad_mfcc(mfcc: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad MFCC along time axis to have exactly max_frames columns."""
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    return mfcc


def main():
    print("=== 01_extract_mfcc.py ===")
    print(f"Base directory: {BASE_DIR}")

    # 1) Load signals
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
    mfcc_list = [
        extract_mfcc(sig, SR, N_MFCC, HOP_LENGTH)
        for sig in all_signals
    ]

    # 3) Determine max frames and pad
    max_frames = max(m.shape[1] for m in mfcc_list)
    print(f"Maximum number of frames: {max_frames}")

    mfcc_padded = np.stack(
        [pad_mfcc(m, max_frames) for m in mfcc_list],
        axis=0
    )
    # Add channel dimension for CNN: (N, 40, T, 1)
    mfcc_padded = np.expand_dims(mfcc_padded, axis=-1)

    print(f"Final MFCC array shape: {mfcc_padded.shape}")
    print(f"Labels shape: {labels.shape}")

    # 4) Save processed arrays
    mfcc_path = PROCESSED_DIR / "mfcc_images.npy"
    labels_path = PROCESSED_DIR / "labels.npy"
    meta_path = PROCESSED_DIR / "mfcc_config.npz"

    np.save(mfcc_path, mfcc_padded)
    np.save(labels_path, labels)
    np.savez(
        meta_path,
        sr=SR,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        max_frames=max_frames,
    )

    print(f"Saved MFCC images to: {mfcc_path}")
    print(f"Saved labels to: {labels_path}")
    print(f"Saved MFCC config to: {meta_path}")


if __name__ == "__main__":
    main()
