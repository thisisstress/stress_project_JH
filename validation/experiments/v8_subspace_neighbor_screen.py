# -*- coding: utf-8 -*-
"""Screen higher-order rank-neighbor subspace ensembles on Train OOF only.

V7 uses every two-feature subspace from eight numerical signals.  This
experiment extends the same leakage-safe idea to dimensions 3 through 8 and
tests whether their different local neighbourhoods improve the frozen V6/V7
predictions.  Test data is never loaded.
"""

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
PAIR_COLUMNS = [
    "mean_working",
    "bmi",
    "cholesterol",
    "height",
    "glucose",
    "weight",
    "cholesterol_glucose_ratio",
    "bone_density",
]
QUANTILES = [0.48, 0.50, 0.52, 0.54, 0.56]
AUX_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def empirical_rank_transform(
    train_values: np.ndarray,
    valid_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_values)
    train_values = np.nan_to_num(train_values, nan=median)
    valid_values = np.nan_to_num(valid_values, nan=median)
    sorted_train = np.sort(train_values)
    train_rank = np.searchsorted(
        sorted_train, train_values, side="right"
    ) / len(train_values)
    valid_rank = np.searchsorted(
        sorted_train, valid_values, side="right"
    ) / len(train_values)
    return train_rank.astype(np.float32), valid_rank.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v7-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path)
    args = parser.parse_args()

    cached = np.load(args.v7_cache)
    target = cached["target"].astype(float)
    v6 = cached["v6"].astype(float)
    pair = cached["aux__pair_all_q52"].astype(float)
    v7 = np.clip(np.round(0.85 * v6 + 0.15 * pair, 2), 0.0, 1.0)

    train = pd.read_csv(args.data_dir / "train.csv")
    if not np.array_equal(train[TARGET].to_numpy(dtype=float), target):
        raise ValueError("V7 cache target does not match train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    features = add_base_features(raw)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(pd.Series(target))
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(raw, bins, groups))
    validate_splits(splits, groups, len(train))

    dimension_oof = {
        dimension: np.full((len(train), len(list(combinations(PAIR_COLUMNS, dimension)))), np.nan)
        for dimension in range(2, len(PAIR_COLUMNS) + 1)
    }

    for fold, (train_index, valid_index) in enumerate(splits):
        train_rank: dict[str, np.ndarray] = {}
        valid_rank: dict[str, np.ndarray] = {}
        for column in PAIR_COLUMNS:
            train_rank[column], valid_rank[column] = empirical_rank_transform(
                features.iloc[train_index][column].to_numpy(dtype=float),
                features.iloc[valid_index][column].to_numpy(dtype=float),
            )
        fold_target = target[train_index]
        for dimension in range(2, len(PAIR_COLUMNS) + 1):
            for subset_index, subset in enumerate(
                combinations(PAIR_COLUMNS, dimension)
            ):
                distance = np.zeros(
                    (len(valid_index), len(train_index)), dtype=np.float32
                )
                for column in subset:
                    distance += np.abs(
                        valid_rank[column][:, None] - train_rank[column][None, :]
                    )
                nearest = np.argmin(distance, axis=1)
                dimension_oof[dimension][valid_index, subset_index] = fold_target[
                    nearest
                ]
        print(f"fold={fold} complete", flush=True)

    auxiliary: dict[str, np.ndarray] = {"pair_v7_q52": pair}
    cumulative: list[np.ndarray] = []
    for dimension, values in dimension_oof.items():
        cumulative.append(values)
        for quantile in QUANTILES:
            q_name = int(round(100 * quantile))
            auxiliary[f"dim{dimension}_q{q_name}"] = np.quantile(
                values, quantile, axis=1
            )
            pooled = np.column_stack(cumulative)
            auxiliary[f"pool2to{dimension}_q{q_name}"] = np.quantile(
                pooled, quantile, axis=1
            )

    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "auxiliary": "pair_v7_q52",
            "aux_weight": 0.15,
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]
    prediction_cache: dict[str, np.ndarray] = {
        "target": target,
        "v6": v6,
        "v7": v7,
        **{f"aux__{name}": values for name, values in auxiliary.items()},
    }
    for name, values in auxiliary.items():
        for weight in AUX_WEIGHTS:
            prediction = np.clip(
                np.round((1.0 - weight) * v6 + weight * values, 2),
                0.0,
                1.0,
            )
            mae = mean_absolute_error(target, prediction)
            candidate = f"{name}__w{weight:.2f}"
            rows.append(
                {
                    "candidate": candidate,
                    "auxiliary": name,
                    "aux_weight": weight,
                    "mae": mae,
                    "delta_vs_v7": mae - v7_mae,
                }
            )
            prediction_cache[f"prediction__{candidate}"] = prediction

    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    if args.cache_output is not None:
        args.cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache_output, **prediction_cache)
    print(f"V7 reference: {v7_mae:.6f}")
    print(summary.head(60).to_string(index=False))


if __name__ == "__main__":
    main()
