# -*- coding: utf-8 -*-
"""Fold-local meta learners over the 28 pair-neighbour target signals.

For every outer fold, training-row pair features are computed leave-group-out
and validation-row features are computed from that outer training fold only.
The meta learner therefore never sees a validation target or a row's own
target through its nearest-neighbour features.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

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
META_SEED = 81173
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
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


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


def augment_pair_matrix(values: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(values, [0.25, 0.40, 0.48, 0.50, 0.52, 0.60, 0.75], axis=1).T
    summary = np.column_stack(
        [
            quantiles,
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
        ]
    )
    return np.column_stack([values, summary])


def build_models() -> dict[str, object]:
    return {
        "quantile_a0p001": make_pipeline(
            RobustScaler(),
            QuantileRegressor(quantile=0.50, alpha=0.001, solver="highs"),
        ),
        "quantile_a0p003": make_pipeline(
            RobustScaler(),
            QuantileRegressor(quantile=0.50, alpha=0.003, solver="highs"),
        ),
        "hist_leaf31": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=5.0,
            random_state=META_SEED,
        ),
        "gbr_depth1": GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=150,
            learning_rate=0.03,
            max_depth=1,
            min_samples_leaf=40,
            random_state=META_SEED,
        ),
        "rf_leaf30": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=30,
            max_features=0.7,
            random_state=META_SEED,
            n_jobs=1,
        ),
        "et_leaf30": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=30,
            max_features=0.7,
            random_state=META_SEED,
            n_jobs=1,
        ),
    }


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

    model_oof = {
        name: np.full(len(train), np.nan) for name in build_models()
    }
    for fold, (train_index, valid_index) in enumerate(splits):
        train_rank: dict[str, np.ndarray] = {}
        valid_rank: dict[str, np.ndarray] = {}
        for column in PAIR_COLUMNS:
            train_rank[column], valid_rank[column] = empirical_rank_transform(
                features.iloc[train_index][column].to_numpy(dtype=float),
                features.iloc[valid_index][column].to_numpy(dtype=float),
            )

        subsets = list(combinations(PAIR_COLUMNS, 2))
        train_pair = np.empty((len(train_index), len(subsets)), dtype=float)
        valid_pair = np.empty((len(valid_index), len(subsets)), dtype=float)
        fold_target = target[train_index]
        fold_groups = groups.iloc[train_index].to_numpy()
        same_group = fold_groups[:, None] == fold_groups[None, :]
        for pair_index, (first, second) in enumerate(subsets):
            train_distance = (
                np.abs(train_rank[first][:, None] - train_rank[first][None, :])
                + np.abs(train_rank[second][:, None] - train_rank[second][None, :])
            )
            train_distance[same_group] = np.inf
            train_nearest = np.argmin(train_distance, axis=1)
            train_pair[:, pair_index] = fold_target[train_nearest]

            valid_distance = (
                np.abs(valid_rank[first][:, None] - train_rank[first][None, :])
                + np.abs(valid_rank[second][:, None] - train_rank[second][None, :])
            )
            valid_nearest = np.argmin(valid_distance, axis=1)
            valid_pair[:, pair_index] = fold_target[valid_nearest]

        meta_train = augment_pair_matrix(train_pair)
        meta_valid = augment_pair_matrix(valid_pair)
        for model_name, model in build_models().items():
            model.fit(meta_train, fold_target)
            model_oof[model_name][valid_index] = np.clip(
                model.predict(meta_valid), 0.0, 1.0
            )
        print(f"fold={fold} complete", flush=True)

    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "meta_model": "none",
            "anchor": "v7",
            "blend_weight": 0.0,
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]
    prediction_cache: dict[str, np.ndarray] = {
        "target": target,
        "v6": v6,
        "v7": v7,
    }
    for model_name, meta_prediction in model_oof.items():
        prediction_cache[f"meta__{model_name}"] = meta_prediction
        for anchor_name, anchor in {"v6": v6, "v7": v7}.items():
            for weight in BLEND_WEIGHTS:
                prediction = np.clip(
                    np.round(
                        (1.0 - weight) * anchor + weight * meta_prediction,
                        2,
                    ),
                    0.0,
                    1.0,
                )
                candidate = f"{anchor_name}__{model_name}__w{weight:.2f}"
                mae = mean_absolute_error(target, prediction)
                rows.append(
                    {
                        "candidate": candidate,
                        "meta_model": model_name,
                        "anchor": anchor_name,
                        "blend_weight": weight,
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
