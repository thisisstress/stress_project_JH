# -*- coding: utf-8 -*-
"""Confirm the fixed V8 greedy recipe on alternate Train-only split seeds."""

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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

from robust_validation_v5 import (
    ID_COL,
    N_SPLITS,
    TARGET,
    add_base_features,
    apply_feature_weights,
    make_duplicate_groups,
    make_preprocessor,
    make_target_bins,
    validate_splits,
)


MODEL_SEED = 42
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
FEATURE_COPY_COUNTS = {
    "mean_working": 1,
    "bmi": 4,
    "cholesterol": 2,
    "height": 2,
    "glucose": 3,
    "weight": 2,
    "cholesterol_glucose_ratio": 2,
    "bone_density": 4,
}


def clipped_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values, 2), 0.0, 1.0)


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


def fold_neighbors(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: np.ndarray,
) -> dict[str, np.ndarray]:
    train_rank: dict[str, np.ndarray] = {}
    valid_rank: dict[str, np.ndarray] = {}
    for column in PAIR_COLUMNS:
        train_rank[column], valid_rank[column] = empirical_rank_transform(
            train_frame[column].to_numpy(dtype=float),
            valid_frame[column].to_numpy(dtype=float),
        )

    matrices: dict[int, np.ndarray] = {}
    triple_distances: np.ndarray | None = None
    for dimension in [2, 3, 4]:
        subsets = list(combinations(PAIR_COLUMNS, dimension))
        targets = np.empty((len(valid_frame), len(subsets)), dtype=float)
        distances = (
            np.empty((len(valid_frame), len(subsets)), dtype=np.float32)
            if dimension == 3
            else None
        )
        for subset_index, subset in enumerate(subsets):
            distance = np.zeros(
                (len(valid_frame), len(train_frame)), dtype=np.float32
            )
            for column in subset:
                distance += np.abs(
                    valid_rank[column][:, None] - train_rank[column][None, :]
                )
            nearest = np.argmin(distance, axis=1)
            targets[:, subset_index] = train_target[nearest]
            if distances is not None:
                distances[:, subset_index] = distance[
                    np.arange(len(valid_frame)), nearest
                ]
        matrices[dimension] = targets
        if dimension == 3:
            triple_distances = distances

    if triple_distances is None:
        raise RuntimeError("Triple distances were not created")
    closest_predictions = {}
    for count in [20, 30]:
        selected = np.argpartition(
            triple_distances, kth=count - 1, axis=1
        )[:, :count]
        selected_targets = np.take_along_axis(
            matrices[3], selected, axis=1
        )
        closest_predictions[f"triple_closest{count}_q50"] = np.quantile(
            selected_targets, 0.50, axis=1
        )
    pool = np.column_stack([matrices[2], matrices[3], matrices[4]])
    return {
        "pair_q52": np.quantile(matrices[2], 0.52, axis=1),
        "dim4_q50": np.quantile(matrices[4], 0.50, axis=1),
        "pool2to4_q48": np.quantile(pool, 0.48, axis=1),
        "pool2to4_q50": np.quantile(pool, 0.50, axis=1),
        **closest_predictions,
    }


def build_recipe(
    v6: np.ndarray,
    pair: np.ndarray,
    squared_mf4_q56: np.ndarray,
    neighbor: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    v7 = clipped_round(0.85 * v6 + 0.15 * pair)
    triple30 = clipped_round(
        0.80 * v7 + 0.20 * neighbor["triple_closest30_q50"]
    )
    squared = clipped_round(0.80 * v7 + 0.20 * squared_mf4_q56)
    pool50 = clipped_round(0.85 * v7 + 0.15 * neighbor["pool2to4_q50"])
    dim4 = clipped_round(0.90 * v7 + 0.10 * neighbor["dim4_q50"])
    pool48 = clipped_round(0.75 * v7 + 0.25 * neighbor["pool2to4_q48"])
    triple20 = clipped_round(
        0.80 * v7 + 0.20 * neighbor["triple_closest20_q50"]
    )

    predictions = {"v7_reference": v7, "greedy_step1": triple30}
    predictions["robust_balanced"] = clipped_round(
        0.825 * v7
        + 0.025 * neighbor["triple_closest30_q50"]
        + 0.050 * neighbor["dim4_q50"]
        + 0.075 * neighbor["pool2to4_q48"]
        + 0.025 * neighbor["pool2to4_q50"]
    )
    current = clipped_round(0.75 * triple30 + 0.25 * squared)
    predictions["greedy_step2"] = current
    current = clipped_round(0.80 * current + 0.20 * pool50)
    predictions["greedy_step3"] = current
    current = clipped_round(0.85 * current + 0.15 * dim4)
    predictions["greedy_step4"] = current
    current = clipped_round(0.80 * current + 0.20 * pool48)
    predictions["greedy_step5"] = current
    current = clipped_round(0.90 * current + 0.10 * triple20)
    predictions["greedy_step6"] = current
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-detail", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--v6-trees", type=int, default=1200)
    parser.add_argument("--criteria-trees", type=int, default=200)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    target = train[TARGET].astype(float).reset_index(drop=True)
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    base = add_base_features(raw)
    weighted = apply_feature_weights(base, FEATURE_COPY_COUNTS)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)
    detail_rows = []

    for split_seed in args.split_seeds:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=split_seed,
        )
        splits = list(splitter.split(raw, bins, groups))
        validate_splits(splits, groups, len(train))
        v6 = np.full(len(train), np.nan)
        squared_mf4_q56 = np.full(len(train), np.nan)
        neighbor_oof = {
            name: np.full(len(train), np.nan)
            for name in [
                "pair_q52",
                "dim4_q50",
                "pool2to4_q48",
                "pool2to4_q50",
                "triple_closest20_q50",
                "triple_closest30_q50",
            ]
        }
        for fold, (train_index, valid_index) in enumerate(splits):
            preprocessor = make_preprocessor(weighted.iloc[train_index])
            train_matrix = preprocessor.fit_transform(weighted.iloc[train_index])
            valid_matrix = preprocessor.transform(weighted.iloc[valid_index])
            v6_model = ExtraTreesRegressor(
                n_estimators=args.v6_trees,
                min_samples_leaf=1,
                max_features=1,
                random_state=MODEL_SEED,
                n_jobs=1,
            )
            v6_model.fit(train_matrix, target.iloc[train_index])
            tree_prediction = np.column_stack(
                [tree.predict(valid_matrix) for tree in v6_model.estimators_]
            )
            v6[valid_index] = clipped_round(
                np.quantile(tree_prediction, 0.52, axis=1)
            )

            criteria_model = ExtraTreesRegressor(
                n_estimators=args.criteria_trees,
                min_samples_leaf=1,
                max_features=4,
                criterion="squared_error",
                random_state=MODEL_SEED,
                n_jobs=1,
            )
            criteria_model.fit(train_matrix, target.iloc[train_index])
            criteria_prediction = np.column_stack(
                [tree.predict(valid_matrix) for tree in criteria_model.estimators_]
            )
            squared_mf4_q56[valid_index] = clipped_round(
                np.quantile(criteria_prediction, 0.56, axis=1)
            )

            neighbor = fold_neighbors(
                base.iloc[train_index],
                base.iloc[valid_index],
                target.iloc[train_index].to_numpy(dtype=float),
            )
            for name, prediction in neighbor.items():
                neighbor_oof[name][valid_index] = prediction
            print(f"seed={split_seed} fold={fold} complete", flush=True)

        predictions = build_recipe(
            v6,
            neighbor_oof["pair_q52"],
            squared_mf4_q56,
            neighbor_oof,
        )
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.cache_dir / f"v8_recipe_seed{split_seed}.npz",
                target=target.to_numpy(dtype=float),
                v6=v6,
                squared_mf4_q56=squared_mf4_q56,
                **{f"neighbor__{name}": values for name, values in neighbor_oof.items()},
                **{f"prediction__{name}": values for name, values in predictions.items()},
            )
        v7_mae = mean_absolute_error(target, predictions["v7_reference"])
        for name, prediction in predictions.items():
            mae = mean_absolute_error(target, prediction)
            detail_rows.append(
                {
                    "split_seed": split_seed,
                    "candidate": name,
                    "mae": mae,
                    "delta_vs_v7": mae - v7_mae,
                }
            )

    detail = pd.DataFrame(detail_rows)
    summary = detail.groupby("candidate", as_index=False).agg(
        mean_mae=("mae", "mean"),
        seed_std=("mae", "std"),
        mean_delta_vs_v7=("delta_vs_v7", "mean"),
        seed_win_rate=("delta_vs_v7", lambda values: float((values < 0).mean())),
        worst_delta=("delta_vs_v7", "max"),
    ).sort_values("mean_mae")
    args.output_detail.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_detail, index=False)
    summary.to_csv(args.output_summary, index=False)
    print(detail.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
