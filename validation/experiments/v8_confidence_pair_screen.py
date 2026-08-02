# -*- coding: utf-8 -*-
"""Train-only V8 screen for confidence-aware neighbor ensembles.

This script keeps V7 as the reference and screens only on the requested
development split seed. Test data is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations, product
from pathlib import Path

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
DEFAULT_SPLIT_SEED = 42
DEFAULT_TREES = 1200
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
PAIR_QUANTILES = [0.48, 0.50, 0.52, 0.54]
CATEGORY_PENALTIES = [0.025, 0.05, 0.10, 0.20]
SCALAR_BLEND_WEIGHTS = np.arange(0.05, 0.301, 0.025)
GATE_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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


def weighted_row_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> np.ndarray:
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, values.shape)
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    cutoff = quantile * sorted_weights.sum(axis=1)
    positions = np.argmax(cumulative >= cutoff[:, None], axis=1)
    return sorted_values[np.arange(len(values)), positions]


def categorical_mismatch(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
) -> np.ndarray:
    columns = train_frame.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    if not columns:
        return np.zeros((len(valid_frame), len(train_frame)), dtype=np.float32)
    train_values = (
        train_frame[columns].fillna("__MISSING__").astype(str).to_numpy()
    )
    valid_values = (
        valid_frame[columns].fillna("__MISSING__").astype(str).to_numpy()
    )
    return np.mean(
        valid_values[:, None, :] != train_values[None, :, :],
        axis=2,
        dtype=np.float32,
    )


def tree_bundle(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: pd.Series,
    n_estimators: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    preprocessor = make_preprocessor(train_frame)
    train_matrix = preprocessor.fit_transform(train_frame)
    valid_matrix = preprocessor.transform(valid_frame)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        max_features=1,
        random_state=MODEL_SEED,
        n_jobs=1,
    )
    model.fit(train_matrix, train_target)
    tree_predictions = np.column_stack(
        [tree.predict(valid_matrix) for tree in model.estimators_]
    )
    q10, q25, q52, q75, q90 = np.quantile(
        tree_predictions,
        [0.10, 0.25, 0.52, 0.75, 0.90],
        axis=1,
    )
    prediction = np.clip(np.round(q52, 2), 0.0, 1.0)
    meta = {
        "tree_iqr": q75 - q25,
        "tree_range80": q90 - q10,
        "tree_std": np.std(tree_predictions, axis=1),
    }
    return prediction, meta


def neighbor_bundle(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_rank: dict[str, np.ndarray] = {}
    valid_rank: dict[str, np.ndarray] = {}
    for column in PAIR_COLUMNS:
        train_rank[column], valid_rank[column] = empirical_rank_transform(
            train_frame[column].to_numpy(dtype=float),
            valid_frame[column].to_numpy(dtype=float),
        )

    mismatch = categorical_mismatch(train_frame, valid_frame)
    pairs = list(combinations(PAIR_COLUMNS, 2))
    pair_targets = np.empty((len(valid_frame), len(pairs)), dtype=np.float64)
    pair_distances = np.empty((len(valid_frame), len(pairs)), dtype=np.float32)
    pair_loo_mae = np.empty(len(pairs), dtype=np.float64)
    categorical_targets = {
        penalty: np.empty_like(pair_targets) for penalty in CATEGORY_PENALTIES
    }

    for pair_index, (first, second) in enumerate(pairs):
        distance = (
            np.abs(valid_rank[first][:, None] - train_rank[first][None, :])
            + np.abs(valid_rank[second][:, None] - train_rank[second][None, :])
        )
        nearest = np.argmin(distance, axis=1)
        pair_targets[:, pair_index] = train_target[nearest]
        pair_distances[:, pair_index] = distance[np.arange(len(valid_frame)), nearest]

        train_distance = (
            np.abs(train_rank[first][:, None] - train_rank[first][None, :])
            + np.abs(train_rank[second][:, None] - train_rank[second][None, :])
        )
        np.fill_diagonal(train_distance, np.inf)
        train_nearest = np.argmin(train_distance, axis=1)
        pair_loo_mae[pair_index] = np.mean(
            np.abs(train_target - train_target[train_nearest])
        )

        for penalty in CATEGORY_PENALTIES:
            nearest_with_category = np.argmin(
                distance + penalty * mismatch,
                axis=1,
            )
            categorical_targets[penalty][:, pair_index] = train_target[
                nearest_with_category
            ]

    candidates: dict[str, np.ndarray] = {}
    for quantile in PAIR_QUANTILES:
        q_name = int(round(quantile * 100))
        candidates[f"pair_all_q{q_name}"] = np.quantile(
            pair_targets, quantile, axis=1
        )

    reliability_order = np.argsort(pair_loo_mae)
    for count in [6, 10, 14, 18, 22]:
        selected = reliability_order[:count]
        for quantile in [0.50, 0.52]:
            q_name = int(round(quantile * 100))
            candidates[f"pair_reliable{count}_q{q_name}"] = np.quantile(
                pair_targets[:, selected], quantile, axis=1
            )

    for count in [6, 10, 14, 18, 22]:
        selected = np.argpartition(
            pair_distances, kth=count - 1, axis=1
        )[:, :count]
        selected_targets = np.take_along_axis(pair_targets, selected, axis=1)
        for quantile in [0.50, 0.52]:
            q_name = int(round(quantile * 100))
            candidates[f"pair_closest{count}_q{q_name}"] = np.quantile(
                selected_targets, quantile, axis=1
            )

    inverse_distance = 1.0 / np.maximum(pair_distances, 1e-4)
    inverse_reliability = 1.0 / np.maximum(pair_loo_mae, 1e-4)
    for quantile in [0.50, 0.52]:
        q_name = int(round(quantile * 100))
        candidates[f"pair_distance_weighted_q{q_name}"] = weighted_row_quantile(
            pair_targets, inverse_distance, quantile
        )
        candidates[f"pair_reliability_weighted_q{q_name}"] = weighted_row_quantile(
            pair_targets, inverse_reliability, quantile
        )

    for penalty, target_matrix in categorical_targets.items():
        penalty_name = str(penalty).replace(".", "p")
        for quantile in [0.50, 0.52]:
            q_name = int(round(quantile * 100))
            candidates[f"pair_cat{penalty_name}_q{q_name}"] = np.quantile(
                target_matrix, quantile, axis=1
            )

    triples = list(combinations(PAIR_COLUMNS, 3))
    triple_targets = np.empty(
        (len(valid_frame), len(triples)), dtype=np.float64
    )
    triple_distances = np.empty(
        (len(valid_frame), len(triples)), dtype=np.float32
    )
    for triple_index, (first, second, third) in enumerate(triples):
        distance = (
            np.abs(valid_rank[first][:, None] - train_rank[first][None, :])
            + np.abs(valid_rank[second][:, None] - train_rank[second][None, :])
            + np.abs(valid_rank[third][:, None] - train_rank[third][None, :])
        )
        nearest = np.argmin(distance, axis=1)
        triple_targets[:, triple_index] = train_target[nearest]
        triple_distances[:, triple_index] = distance[
            np.arange(len(valid_frame)), nearest
        ]

    for quantile in [0.50, 0.52]:
        q_name = int(round(quantile * 100))
        candidates[f"triple_all_q{q_name}"] = np.quantile(
            triple_targets, quantile, axis=1
        )
        candidates[f"pair_triple_pool_q{q_name}"] = np.quantile(
            np.column_stack([pair_targets, triple_targets]),
            quantile,
            axis=1,
        )

    for count in [10, 20, 30, 40]:
        selected = np.argpartition(
            triple_distances, kth=count - 1, axis=1
        )[:, :count]
        selected_targets = np.take_along_axis(
            triple_targets, selected, axis=1
        )
        candidates[f"triple_closest{count}_q50"] = np.quantile(
            selected_targets, 0.50, axis=1
        )
        candidates[f"triple_closest{count}_q52"] = np.quantile(
            selected_targets, 0.52, axis=1
        )

    meta = {
        "pair_std": np.std(pair_targets, axis=1),
        "pair_iqr": np.quantile(pair_targets, 0.75, axis=1)
        - np.quantile(pair_targets, 0.25, axis=1),
        "pair_distance_mean": np.mean(pair_distances, axis=1),
        "pair_distance_min": np.min(pair_distances, axis=1),
        "pair_distance_iqr": np.quantile(pair_distances, 0.75, axis=1)
        - np.quantile(pair_distances, 0.25, axis=1),
        "triple_std": np.std(triple_targets, axis=1),
        "triple_distance_mean": np.mean(triple_distances, axis=1),
    }
    return candidates, meta


def assign_fold_values(
    store: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    valid_index: np.ndarray,
    row_count: int,
) -> None:
    for name, prediction in values.items():
        if name not in store:
            store[name] = np.full(row_count, np.nan, dtype=float)
        store[name][valid_index] = prediction


def evaluate_candidates(
    target: np.ndarray,
    v6: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
) -> pd.DataFrame:
    baseline_pair = auxiliary["pair_all_q52"]
    v7 = np.clip(
        np.round(0.85 * v6 + 0.15 * baseline_pair, 2),
        0.0,
        1.0,
    )
    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "family": "reference",
            "auxiliary": "pair_all_q52",
            "recipe": "weight=0.15;round2",
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]

    for name, aux_prediction in auxiliary.items():
        for weight in SCALAR_BLEND_WEIGHTS:
            prediction = np.clip(
                np.round(
                    (1.0 - weight) * v6 + weight * aux_prediction,
                    2,
                ),
                0.0,
                1.0,
            )
            mae = mean_absolute_error(target, prediction)
            rows.append(
                {
                    "candidate": f"scalar__{name}__w{weight:.3f}",
                    "family": "scalar",
                    "auxiliary": name,
                    "recipe": f"weight={weight:.3f};round2",
                    "mae": mae,
                    "delta_vs_v7": mae - v7_mae,
                }
            )

    signal_values = {
        **meta,
        "pair_tree_disagreement": np.abs(baseline_pair - v6),
    }
    for signal_name, signal in signal_values.items():
        lower, upper = np.quantile(signal, [1.0 / 3.0, 2.0 / 3.0])
        bins = np.where(signal <= lower, 0, np.where(signal <= upper, 1, 2))
        for weights in product(GATE_WEIGHTS, repeat=3):
            row_weight = np.choose(bins, weights)
            prediction = np.clip(
                np.round(
                    (1.0 - row_weight) * v6
                    + row_weight * baseline_pair,
                    2,
                ),
                0.0,
                1.0,
            )
            mae = mean_absolute_error(target, prediction)
            recipe = {
                "signal": signal_name,
                "lower": float(lower),
                "upper": float(upper),
                "weights": list(weights),
            }
            rows.append(
                {
                    "candidate": (
                        f"gate__{signal_name}__"
                        + "_".join(f"{weight:.2f}" for weight in weights)
                    ),
                    "family": "gate",
                    "auxiliary": "pair_all_q52",
                    "recipe": json.dumps(recipe, ensure_ascii=False),
                    "mae": mae,
                    "delta_vs_v7": mae - v7_mae,
                }
            )
    return pd.DataFrame(rows).sort_values(["mae", "candidate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_TREES)
    parser.add_argument("--cache-output", type=Path)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].astype(float).reset_index(drop=True)
    base = add_base_features(raw)
    weighted = apply_feature_weights(base, FEATURE_COPY_COUNTS)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=args.split_seed,
    )
    splits = list(splitter.split(raw, bins, groups))
    validate_splits(splits, groups, len(train))

    v6_oof = np.full(len(train), np.nan, dtype=float)
    auxiliary_oof: dict[str, np.ndarray] = {}
    meta_oof: dict[str, np.ndarray] = {}
    for fold, (train_index, valid_index) in enumerate(splits):
        v6_prediction, tree_meta = tree_bundle(
            weighted.iloc[train_index],
            weighted.iloc[valid_index],
            target.iloc[train_index],
            args.n_estimators,
        )
        neighbor_predictions, neighbor_meta = neighbor_bundle(
            base.iloc[train_index],
            base.iloc[valid_index],
            target.iloc[train_index].to_numpy(),
        )
        v6_oof[valid_index] = v6_prediction
        assign_fold_values(
            auxiliary_oof,
            neighbor_predictions,
            valid_index,
            len(train),
        )
        assign_fold_values(
            meta_oof,
            {**tree_meta, **neighbor_meta},
            valid_index,
            len(train),
        )
        print(f"seed={args.split_seed} fold={fold} complete", flush=True)

    for name, values in {
        "v6": v6_oof,
        **auxiliary_oof,
        **meta_oof,
    }.items():
        if np.isnan(values).any():
            raise RuntimeError(f"OOF prediction contains NaN: {name}")

    summary = evaluate_candidates(
        target.to_numpy(),
        v6_oof,
        auxiliary_oof,
        meta_oof,
    )
    output = args.results_dir / (
        f"v8_confidence_pair_screen_seed{args.split_seed}_summary.csv"
    )
    summary.to_csv(output, index=False)
    if args.cache_output is not None:
        args.cache_output.parent.mkdir(parents=True, exist_ok=True)
        cache_values = {
            "target": target.to_numpy(),
            "v6": v6_oof,
        }
        cache_values.update(
            {f"aux__{name}": value for name, value in auxiliary_oof.items()}
        )
        cache_values.update(
            {f"meta__{name}": value for name, value in meta_oof.items()}
        )
        np.savez_compressed(args.cache_output, **cache_values)
        print(f"Saved cache: {args.cache_output}")
    print(summary.head(40).to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
