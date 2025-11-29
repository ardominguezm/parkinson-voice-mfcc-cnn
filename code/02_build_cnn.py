#!/usr/bin/env python3
"""
02_build_cnn.py

Defines the CNN architecture used in the study and optionally prints
the model summary using the saved MFCC images for input shape.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def build_cnn_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """
    Build the convolutional neural network.

    input_shape: (height, width, channels) = (40, max_frames, 1)
    """
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


def main():
    print("=== 02_build_cnn.py ===")
    mfcc_path = PROCESSED_DIR / "mfcc_images.npy"

    if not mfcc_path.exists():
        raise FileNotFoundError(
            f"MFCC images not found at {mfcc_path}. "
            "Run 01_extract_mfcc.py first."
        )

    mfcc_images = np.load(mfcc_path)
    input_shape = mfcc_images.shape[1:]
    print(f"Loaded MFCC images with shape: {mfcc_images.shape}")
    print(f"Input shape for CNN: {input_shape}")

    model = build_cnn_model(input_shape)
    model.summary()


if __name__ == "__main__":
    main()
