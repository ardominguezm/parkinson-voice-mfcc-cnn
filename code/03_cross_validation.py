#!/usr/bin/env python3
"""
03_cross_validation.py

Performs 10-fold stratified cross-validation using the CNN model
on MFCC spectral images and computes accuracy, F1, precision,
recall, and ROC AUC, including mean, std, and coefficient of variation.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def build_cnn_model(input_shape):
    """Same architecture as in 02_build_cnn.py."""
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
    print("=== 03_cross_validation.py ===")

    mfcc_path = PROCESSED_DIR / "mfcc_images.npy"
    labels_path = PROCESSED_DIR / "labels.npy"

    if not mfcc_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "MFCC images or labels not found. "
            "Run 01_extract_mfcc.py first."
        )

    X = np.load(mfcc_path)
    y = np.load(labels_path)

    print(f"Loaded MFCC images: {X.shape}")
    print(f"Loaded labels: {y.shape}")

    input_shape = X.shape[1:]
    k = 10

    skf = StratifiedKFold(
        n_splits=k,
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
        print(f"\nFold {fold_idx}/{k}")
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
    csv_path = ANALYSIS_DIR / "cv_metrics_real_data.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved fold-wise metrics to: {csv_path}")

    # Summary statistics
    print("\n--- Cross-validation summary statistics ---")
    summary = {}
    for col in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        vals = df_metrics[col].values
        mean_val = vals.mean()
        std_val = vals.std()
        cv = std_val / mean_val if mean_val != 0 else 0.0
        summary[col] = (mean_val, std_val, cv)
        print(
            f"{col.capitalize()}: {mean_val:.4f} ± {std_val:.4f} "
            f"(CV = {cv:.4f})"
        )

    # Also store summary as CSV
    summary_rows = []
    for metric, (mean_val, std_val, cv) in summary.items():
        summary_rows.append({
            "metric": metric,
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = ANALYSIS_DIR / "cv_metrics_real_data_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    main()
