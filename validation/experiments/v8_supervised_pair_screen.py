# -*- coding: utf-8 -*-
"""Train-only tuning of per-pair distance ratios and neighbor counts."""

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
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in [VALIDATION_DIR, EXPERIMENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from v8_confidence_pair_screen import (
    PAIR_COLUMNS,
    empirical_rank_transform,
    weighted_row_quantile,
)


SCREEN_SEED = 42
TUNING_SEED = 83059
LAMBDA_VALUES = [0.25, 0.50, 1.00, 2.00, 4.00]
K_VALUES = [1, 3, 5]
QUANTILES = [0.48, 0.50, 0.52, 0.54]
BLEND_WEIGHTS = np.arange(0.05, 0.401, 0.025)


def median_neighbor_target(
    distance: np.ndarray,
    target: np.ndarray,
    k: int,
) -> np.ndarray:
    if k == 1:
        return target[np.argmin(distance, axis=1)]
    index = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    return np.median(target[index], axis=1)


def tuned_pair_predictions(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: np.ndarray,
    tune_size: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    train_rank: dict[str, np.ndarray] = {}
    valid_rank: dict[str, np.ndarray] = {}
    for column in PAIR_COLUMNS:
        train_rank[column], valid_rank[column] = empirical_rank_transform(
            train_frame[column].to_numpy(dtype=float),
            valid_frame[column].to_numpy(dtype=float),
        )

    rng = np.random.default_rng(TUNING_SEED + fold)
    tune_size = min(tune_size, len(train_frame))
    tune_index = np.sort(
        rng.choice(len(train_frame), size=tune_size, replace=False)
    )
    pairs = list(combinations(PAIR_COLUMNS, 2))
    tuned_targets = np.empty((len(valid_frame), len(pairs)), dtype=float)
    selected_errors = np.empty(len(pairs), dtype=float)
    selected_rows = []

    for pair_index, (first, second) in enumerate(pairs):
        tune_first = np.abs(
            train_rank[first][tune_index, None] - train_rank[first][None, :]
        )
        tune_second = np.abs(
            train_rank[second][tune_index, None] - train_rank[second][None, :]
        )
        best = None
        for distance_ratio in LAMBDA_VALUES:
            tune_distance = distance_ratio * tune_first + tune_second
            tune_distance[np.arange(tune_size), tune_index] = np.inf
            for k in K_VALUES:
                tune_prediction = median_neighbor_target(
                    tune_distance, train_target, k
                )
                mae = mean_absolute_error(
                    train_target[tune_index], tune_prediction
                )
                candidate = (mae, abs(np.log2(distance_ratio)), k, distance_ratio)
                if best is None or candidate < best:
                    best = candidate

        assert best is not None
        best_mae, _, best_k, best_ratio = best
        valid_distance = (
            best_ratio
            * np.abs(
                valid_rank[first][:, None] - train_rank[first][None, :]
            )
            + np.abs(
                valid_rank[second][:, None] - train_rank[second][None, :]
            )
        )
        tuned_targets[:, pair_index] = median_neighbor_target(
            valid_distance, train_target, best_k
        )
        selected_errors[pair_index] = best_mae
        selected_rows.append(
            {
                "fold": fold,
                "first": first,
                "second": second,
                "distance_ratio": best_ratio,
                "k": best_k,
                "tuning_mae": best_mae,
            }
        )
    return tuned_targets, selected_errors, selected_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v7-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune-size", type=int, default=800)
    args = parser.parse_args()

    cache = np.load(args.v7_cache)
    cached_target = cache["target"].astype(float)
    v6 = cache["v6"].astype(float)
    pair_v7 = cache["aux__pair_all_q52"].astype(float)
    v7 = np.clip(np.round(0.85 * v6 + 0.15 * pair_v7, 2), 0.0, 1.0)

    train = pd.read_csv(args.data_dir / "train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].astype(float).reset_index(drop=True)
    if not np.array_equal(target.to_numpy(), cached_target):
        raise ValueError("V7 cache target does not match train.csv")
    base = add_base_features(raw)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(raw, bins, groups))
    validate_splits(splits, groups, len(train))

    candidate_oof: dict[str, np.ndarray] = {}
    selected_rows = []
    for fold, (train_index, valid_index) in enumerate(splits):
        target_matrix, pair_errors, fold_rows = tuned_pair_predictions(
            base.iloc[train_index],
            base.iloc[valid_index],
            target.iloc[train_index].to_numpy(),
            args.tune_size,
            fold,
        )
        selected_rows.extend(fold_rows)
        inverse_error = 1.0 / np.maximum(pair_errors, 1e-4)
        for quantile in QUANTILES:
            q_name = int(round(quantile * 100))
            values = {
                f"supervised_pair_q{q_name}": np.quantile(
                    target_matrix, quantile, axis=1
                ),
                f"supervised_pair_weighted_q{q_name}": weighted_row_quantile(
                    target_matrix, inverse_error, quantile
                ),
            }
            for name, prediction in values.items():
                if name not in candidate_oof:
                    candidate_oof[name] = np.full(len(train), np.nan)
                candidate_oof[name][valid_index] = prediction
        print(f"fold={fold} complete", flush=True)

    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "auxiliary": "pair_all_q52",
            "blend_weight": 0.15,
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]
    for auxiliary_name, auxiliary in candidate_oof.items():
        for weight in BLEND_WEIGHTS:
            prediction = np.clip(
                np.round((1.0 - weight) * v6 + weight * auxiliary, 2),
                0.0,
                1.0,
            )
            mae = mean_absolute_error(target, prediction)
            rows.append(
                {
                    "candidate": f"{auxiliary_name}__w{weight:.3f}",
                    "auxiliary": auxiliary_name,
                    "blend_weight": weight,
                    "mae": mae,
                    "delta_vs_v7": mae - v7_mae,
                }
            )
    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    pd.DataFrame(selected_rows).to_csv(
        args.output.with_name(args.output.stem + "_selected.csv"), index=False
    )
    print(f"V7 reference: {v7_mae:.6f}")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
