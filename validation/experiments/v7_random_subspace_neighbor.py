# -*- coding: utf-8 -*-
"""Pairwise random-subspace nearest-neighbor ensemble, Train-only screen."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
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
QUANTILES = [0.48, 0.50, 0.52, 0.54]
K_VALUES = [1, 3]
FEATURE_SETS = {
    "mi5": [
        "weight",
        "bone_density",
        "height",
        "glucose",
        "cholesterol",
    ],
    "v6_8": [
        "mean_working",
        "bmi",
        "cholesterol",
        "height",
        "glucose",
        "weight",
        "cholesterol_glucose_ratio",
        "bone_density",
    ],
    "original9": [
        "age",
        "height",
        "weight",
        "cholesterol",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "glucose",
        "bone_density",
        "mean_working",
    ],
    "all_numeric": [
        "age",
        "height",
        "weight",
        "cholesterol",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "glucose",
        "bone_density",
        "mean_working",
        "bmi",
        "pulse_pressure",
        "mean_arterial_pressure",
        "blood_pressure_ratio",
        "cholesterol_glucose_ratio",
        "missing_count",
    ],
}


def empirical_rank_transform(
    train_values: np.ndarray,
    valid_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_values)
    train_values = np.nan_to_num(train_values, nan=median)
    valid_values = np.nan_to_num(valid_values, nan=median)
    sorted_values = np.sort(train_values)
    train_rank = np.searchsorted(
        sorted_values, train_values, side="right"
    ) / len(train_values)
    valid_rank = np.searchsorted(
        sorted_values, valid_values, side="right"
    ) / len(train_values)
    return train_rank.astype(np.float32), valid_rank.astype(np.float32)


def pair_predictions(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: np.ndarray,
    columns: list[str],
    k: int,
) -> np.ndarray:
    train_rank = {}
    valid_rank = {}
    for column in columns:
        train_rank[column], valid_rank[column] = empirical_rank_transform(
            train_frame[column].to_numpy(dtype=float),
            valid_frame[column].to_numpy(dtype=float),
        )
    predictions = []
    for first, second in combinations(columns, 2):
        distance = (
            np.abs(valid_rank[first][:, None] - train_rank[first][None, :])
            + np.abs(valid_rank[second][:, None] - train_rank[second][None, :])
        )
        neighbor_index = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
        predictions.append(np.median(train_target[neighbor_index], axis=1))
    return np.column_stack(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    raw = train.drop(columns=[TARGET, ID_COL])
    features = add_base_features(raw)
    target = train[TARGET].astype(float)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(features, bins, groups))
    validate_splits(splits, groups, len(train))

    prediction_store: dict[str, np.ndarray] = {}
    for set_name, columns in FEATURE_SETS.items():
        for k in K_VALUES:
            matrices = []
            for train_index, valid_index in splits:
                matrix = pair_predictions(
                    features.iloc[train_index],
                    features.iloc[valid_index],
                    target.iloc[train_index].to_numpy(),
                    columns,
                    k,
                )
                matrices.append((valid_index, matrix))
            for quantile in QUANTILES:
                name = f"pair_{set_name}_k{k}_q{int(quantile * 100):02d}"
                prediction_store[name] = np.full(len(train), np.nan)
                for valid_index, matrix in matrices:
                    prediction_store[name][valid_index] = np.quantile(
                        matrix, quantile, axis=1
                    )
            print(f"{set_name} k={k} complete", flush=True)

    rows = []
    oof_rows = []
    for name, raw_prediction in prediction_store.items():
        for variant, prediction in {
            "raw": np.clip(raw_prediction, 0.0, 1.0),
            "round2": np.clip(np.round(raw_prediction, 2), 0.0, 1.0),
        }.items():
            candidate = f"{name}_{variant}"
            rows.append(
                {
                    "candidate": candidate,
                    "mean_mae": mean_absolute_error(target, prediction),
                }
            )
            oof_rows.extend(
                {
                    "split_seed": SCREEN_SEED,
                    "row": int(row),
                    "candidate": candidate,
                    "target": float(target.iloc[row]),
                    "prediction": float(prediction[row]),
                }
                for row in range(len(train))
            )
    summary = pd.DataFrame(rows).sort_values("mean_mae")
    summary.to_csv(
        args.results_dir / "v7_pair_neighbor_seed42_summary.csv", index=False
    )
    pd.DataFrame(oof_rows).to_csv(
        args.results_dir / "v7_pair_neighbor_seed42_oof.csv", index=False
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
