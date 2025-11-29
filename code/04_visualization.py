#!/usr/bin/env python3
"""
04_visualization.py

Visualizes MFCC spectral images for one healthy control and one
Parkinson's disease subject, and saves the figure to analysis/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== 04_visualization.py ===")

    mfcc_path = PROCESSED_DIR / "mfcc_images.npy"
    labels_path = PROCESSED_DIR / "labels.npy"

    if not mfcc_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "MFCC images or labels not found. "
            "Run 01_extract_mfcc.py first."
        )

    mfcc_images = np.load(mfcc_path)
    labels = np.load(labels_path)

    print(f"MFCC images shape: {mfcc_images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Find index for control (0) and PD (1)
    idx_control_candidates = np.where(labels == 0)[0]
    idx_pd_candidates = np.where(labels == 1)[0]

    if len(idx_control_candidates) == 0 or len(idx_pd_candidates) == 0:
        raise ValueError("Not enough samples for each class to visualize.")

    idx_control = int(idx_control_candidates[0])
    idx_pd = int(idx_pd_candidates[0])

    img_control = mfcc_images[idx_control].squeeze()  # (40, T)
    img_pd = mfcc_images[idx_pd].squeeze()            # (40, T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(
        img_control,
        aspect="auto",
        origin="lower"
    )
    axes[0].set_title("MFCC Spectrum - Healthy Control")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("MFCC Coefficients")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        img_pd,
        aspect="auto",
        origin="lower"
    )
    axes[1].set_title("MFCC Spectrum - Parkinson's Disease")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("MFCC Coefficients")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = ANALYSIS_DIR / "mfcc_example_control_vs_pd.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Saved MFCC comparison figure to: {fig_path}")


if __name__ == "__main__":
    main()
