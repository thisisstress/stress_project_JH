# -*- coding: utf-8 -*-
"""Screen multi-seed V6 tree pools while keeping the V7 pair signal fixed."""

from __future__ import annotations

import argparse
import os
import sys
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


SCREEN_SEED = 42
MODEL_SEEDS = [42, 7, 2026, 7777]
QUANTILES = [0.48, 0.50, 0.51, 0.52, 0.53, 0.54, 0.56]
TREE_MIX_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
PAIR_WEIGHTS = [0.10, 0.15, 0.20, 0.25]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v7-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees-per-seed", type=int, default=600)
    args = parser.parse_args()

    cache = np.load(args.v7_cache)
    cached_target = cache["target"].astype(float)
    cached_v6 = cache["v6"].astype(float)
    pair = cache["aux__pair_all_q52"].astype(float)
    v7 = np.clip(np.round(0.85 * cached_v6 + 0.15 * pair, 2), 0.0, 1.0)

    train = pd.read_csv(args.data_dir / "train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].astype(float).reset_index(drop=True)
    if not np.array_equal(target.to_numpy(), cached_target):
        raise ValueError("V7 cache target does not match train.csv")
    weighted = apply_feature_weights(add_base_features(raw), FEATURE_COPY_COUNTS)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(raw, bins, groups))
    validate_splits(splits, groups, len(train))

    tree_candidate_oof: dict[str, np.ndarray] = {}
    for fold, (train_index, valid_index) in enumerate(splits):
        preprocessor = make_preprocessor(weighted.iloc[train_index])
        train_matrix = preprocessor.fit_transform(weighted.iloc[train_index])
        valid_matrix = preprocessor.transform(weighted.iloc[valid_index])
        seed_tree_predictions = {}
        for model_seed in MODEL_SEEDS:
            model = ExtraTreesRegressor(
                n_estimators=args.trees_per_seed,
                min_samples_leaf=1,
                max_features=1,
                random_state=model_seed,
                n_jobs=1,
            )
            model.fit(train_matrix, target.iloc[train_index])
            seed_tree_predictions[model_seed] = np.column_stack(
                [tree.predict(valid_matrix) for tree in model.estimators_]
            )

        pool_specs = {
            "seed42": [42],
            "pool2": [42, 7],
            "pool3": [42, 7, 2026],
            "pool4": MODEL_SEEDS,
        }
        for pool_name, seeds in pool_specs.items():
            pooled = np.column_stack(
                [seed_tree_predictions[seed] for seed in seeds]
            )
            for quantile in QUANTILES:
                q_name = int(round(quantile * 100))
                name = f"{pool_name}_q{q_name}"
                if name not in tree_candidate_oof:
                    tree_candidate_oof[name] = np.full(len(train), np.nan)
                tree_candidate_oof[name][valid_index] = np.clip(
                    np.round(np.quantile(pooled, quantile, axis=1), 2),
                    0.0,
                    1.0,
                )

        for quantile in QUANTILES:
            q_name = int(round(quantile * 100))
            per_seed = np.column_stack(
                [
                    np.quantile(seed_tree_predictions[seed], quantile, axis=1)
                    for seed in MODEL_SEEDS
                ]
            )
            for aggregate, values in {
                "seed_quantile_mean": np.mean(per_seed, axis=1),
                "seed_quantile_median": np.median(per_seed, axis=1),
            }.items():
                name = f"{aggregate}_q{q_name}"
                if name not in tree_candidate_oof:
                    tree_candidate_oof[name] = np.full(len(train), np.nan)
                tree_candidate_oof[name][valid_index] = np.clip(
                    np.round(values, 2), 0.0, 1.0
                )
        print(f"fold={fold} complete", flush=True)

    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "tree_candidate": "cached_v6",
            "tree_mix_weight": 0.0,
            "pair_weight": 0.15,
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]
    for tree_name, tree_prediction in tree_candidate_oof.items():
        for tree_mix_weight in TREE_MIX_WEIGHTS:
            mixed_tree = np.clip(
                np.round(
                    (1.0 - tree_mix_weight) * cached_v6
                    + tree_mix_weight * tree_prediction,
                    2,
                ),
                0.0,
                1.0,
            )
            for pair_weight in PAIR_WEIGHTS:
                prediction = np.clip(
                    np.round(
                        (1.0 - pair_weight) * mixed_tree
                        + pair_weight * pair,
                        2,
                    ),
                    0.0,
                    1.0,
                )
                mae = mean_absolute_error(target, prediction)
                rows.append(
                    {
                        "candidate": (
                            f"{tree_name}__tm{tree_mix_weight:.2f}"
                            f"__pw{pair_weight:.2f}"
                        ),
                        "tree_candidate": tree_name,
                        "tree_mix_weight": tree_mix_weight,
                        "pair_weight": pair_weight,
                        "mae": mae,
                        "delta_vs_v7": mae - v7_mae,
                    }
                )
    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"V7 reference: {v7_mae:.6f}")
    print(summary.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
