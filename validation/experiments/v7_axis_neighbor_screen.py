# -*- coding: utf-8 -*-
"""Axis-wise nearest-neighbor models using Train-only folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

from robust_validation_v5 import (
    ID_COL,
    N_SPLITS,
    TARGET,
    add_base_features,
    make_duplicate_groups,
    make_target_bins,
    validate_splits,
)


SCREEN_SEED = 42
K_VALUES = [1, 3, 5, 9]
QUANTILES = [0.48, 0.50, 0.52, 0.54]


def weighted_row_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> np.ndarray:
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, axis=1)
    cutoff = quantile * weights.sum()
    positions = np.argmax(cumulative >= cutoff, axis=1)
    return sorted_values[np.arange(len(values)), positions]


def axis_neighbor_predictions(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    numeric = train_frame.select_dtypes(include=["number"]).columns.tolist()
    predictions = []
    for column in numeric:
        train_values = train_frame[column].to_numpy(dtype=float)
        valid_values = valid_frame[column].to_numpy(dtype=float)
        median = np.nanmedian(train_values)
        train_values = np.nan_to_num(train_values, nan=median)
        valid_values = np.nan_to_num(valid_values, nan=median)
        distance = np.abs(valid_values[:, None] - train_values[None, :])
        neighbor_index = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
        neighbor_target = train_target[neighbor_index]
        predictions.append(np.median(neighbor_target, axis=1))
    return np.column_stack(predictions), numeric


def feature_weights(columns: list[str], profile: str) -> np.ndarray:
    weights = {column: 1.0 for column in columns}
    if profile == "v6":
        weights.update(
            {
                "mean_working": 2.0,
                "bmi": 5.0,
                "cholesterol": 3.0,
                "height": 3.0,
                "glucose": 4.0,
                "weight": 3.0,
                "cholesterol_glucose_ratio": 3.0,
                "bone_density": 5.0,
            }
        )
    elif profile == "mi":
        weights.update(
            {
                "weight": 5.0,
                "bone_density": 5.0,
                "height": 5.0,
                "glucose": 4.0,
                "cholesterol": 4.0,
                "systolic_blood_pressure": 3.0,
                "diastolic_blood_pressure": 2.0,
                "bmi": 3.0,
                "cholesterol_glucose_ratio": 3.0,
            }
        )
    return np.array([weights[column] for column in columns], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL])
    features = add_base_features(raw_features)
    target = train[TARGET].astype(float)
    groups = make_duplicate_groups(raw_features)
    bins = make_target_bins(target)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(features, bins, groups))
    validate_splits(splits, groups, len(train))

    prediction_store: dict[str, np.ndarray] = {}
    for k in K_VALUES:
        fold_matrices: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
        for train_index, valid_index in splits:
            matrix, columns = axis_neighbor_predictions(
                features.iloc[train_index],
                features.iloc[valid_index],
                target.iloc[train_index].to_numpy(),
                k,
            )
            fold_matrices.append((valid_index, matrix, columns))

        for profile in ["equal", "v6", "mi"]:
            for quantile in QUANTILES:
                name = f"axis_k{k}_{profile}_q{int(quantile * 100):02d}"
                prediction_store[name] = np.full(len(train), np.nan)
                for valid_index, matrix, columns in fold_matrices:
                    weights = feature_weights(columns, profile)
                    prediction_store[name][valid_index] = weighted_row_quantile(
                        matrix,
                        weights,
                        quantile,
                    )

    rows = []
    summary_rows = []
    for name, raw_prediction in prediction_store.items():
        for variant, prediction in {
            "raw": np.clip(raw_prediction, 0.0, 1.0),
            "round2": np.clip(np.round(raw_prediction, 2), 0.0, 1.0),
        }.items():
            candidate = f"{name}_{variant}"
            mae = mean_absolute_error(target, prediction)
            summary_rows.append({"candidate": candidate, "mean_mae": mae})
            rows.extend(
                {
                    "split_seed": SCREEN_SEED,
                    "row": int(row),
                    "candidate": candidate,
                    "target": float(target.iloc[row]),
                    "prediction": float(prediction[row]),
                }
                for row in range(len(train))
            )

    summary = pd.DataFrame(summary_rows).sort_values("mean_mae")
    summary.to_csv(
        args.results_dir / "v7_axis_neighbor_seed42_summary.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        args.results_dir / "v7_axis_neighbor_seed42_oof.csv", index=False
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
