# -*- coding: utf-8 -*-
"""Confirm and audit the fixed V7 pair-neighbor blend.

Fixed before confirmation:
- V6 adaptive feature-probability ExtraTrees q52 round2: 85%
- pairwise empirical-rank 1-NN over the fixed V6 eight-feature set q52: 15%
- final prediction rounded to 0.01
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
N_ESTIMATORS = 1200
PAIR_WEIGHT = 0.15
PAIR_QUANTILE = 0.52
REFERENCE_NAME = "work1_bmi4_bone4_glucose3_q520_round2"
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
AUDIT3_SEEDS = [232789, 257053, 279497]
BOOTSTRAP_SEED = 20260803


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


def pair_prediction(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: np.ndarray,
) -> np.ndarray:
    train_rank = {}
    valid_rank = {}
    for column in PAIR_COLUMNS:
        train_rank[column], valid_rank[column] = empirical_rank_transform(
            train_frame[column].to_numpy(dtype=float),
            valid_frame[column].to_numpy(dtype=float),
        )
    pair_values = []
    for first, second in combinations(PAIR_COLUMNS, 2):
        distance = (
            np.abs(valid_rank[first][:, None] - train_rank[first][None, :])
            + np.abs(valid_rank[second][:, None] - train_rank[second][None, :])
        )
        nearest = np.argmin(distance, axis=1)
        pair_values.append(train_target[nearest])
    return np.quantile(
        np.column_stack(pair_values), PAIR_QUANTILE, axis=1
    )


def v6_prediction(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_target: pd.Series,
) -> np.ndarray:
    preprocessor = make_preprocessor(train_frame)
    train_matrix = preprocessor.fit_transform(train_frame)
    valid_matrix = preprocessor.transform(valid_frame)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=1,
        max_features=1,
        random_state=MODEL_SEED,
        n_jobs=1,
    )
    model.fit(train_matrix, train_target)
    tree_predictions = np.column_stack(
        [tree.predict(valid_matrix) for tree in model.estimators_]
    )
    return np.clip(
        np.round(np.quantile(tree_predictions, 0.52, axis=1), 2),
        0.0,
        1.0,
    )


def cluster_interval(
    row_delta: np.ndarray,
    groups: pd.Series,
    draws: int = 10_000,
) -> tuple[float, float]:
    table = pd.DataFrame(
        {"group": groups.to_numpy(), "delta": row_delta}
    ).groupby("group", as_index=False).agg(
        delta_sum=("delta", "sum"),
        row_count=("delta", "size"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(draws)
    count = len(table)
    sums = table["delta_sum"].to_numpy()
    sizes = table["row_count"].to_numpy()
    for draw in range(draws):
        selected = rng.integers(0, count, size=count)
        estimates[draw] = sums[selected].sum() / sizes[selected].sum()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["confirm", "audit3"], required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--reference-oof", type=Path)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or (AUDIT3_SEEDS if args.stage == "audit3" else [2026, 3407])
    train = pd.read_csv(args.data_dir / "train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].astype(float).reset_index(drop=True)
    base = add_base_features(raw)
    weighted = apply_feature_weights(base, FEATURE_COPY_COUNTS)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)

    cached_reference = None
    if args.reference_oof is not None:
        cached_reference = pd.read_csv(args.reference_oof)
        cached_reference = cached_reference.loc[
            cached_reference["candidate"] == REFERENCE_NAME
        ]

    oof_rows = []
    detail_rows = []
    row_deltas = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(raw, bins, groups))
        validate_splits(splits, groups, len(train))
        pair_oof = np.full(len(train), np.nan)
        v6_oof = np.full(len(train), np.nan)
        if cached_reference is not None:
            cached_seed = cached_reference.loc[
                cached_reference["split_seed"] == seed
            ].sort_values("row")
            if len(cached_seed) != len(train):
                raise ValueError(f"Reference OOF missing seed {seed}")
            v6_oof = cached_seed["prediction"].to_numpy()

        for fold, (train_index, valid_index) in enumerate(splits):
            pair_oof[valid_index] = pair_prediction(
                base.iloc[train_index],
                base.iloc[valid_index],
                target.iloc[train_index].to_numpy(),
            )
            if cached_reference is None:
                v6_oof[valid_index] = v6_prediction(
                    weighted.iloc[train_index],
                    weighted.iloc[valid_index],
                    target.iloc[train_index],
                )
            print(
                f"stage={args.stage} seed={seed} fold={fold} complete",
                flush=True,
            )

        blend = np.clip(
            np.round(
                (1.0 - PAIR_WEIGHT) * v6_oof + PAIR_WEIGHT * pair_oof,
                2,
            ),
            0.0,
            1.0,
        )
        v6_error = np.abs(target.to_numpy() - v6_oof)
        blend_error = np.abs(target.to_numpy() - blend)
        delta = blend_error - v6_error
        row_deltas.append(delta)
        detail_rows.append(
            {
                "stage": args.stage,
                "split_seed": seed,
                "v6_mae": v6_error.mean(),
                "pair_mae": mean_absolute_error(target, pair_oof),
                "blend_mae": blend_error.mean(),
                "delta_vs_v6": delta.mean(),
            }
        )
        for candidate, prediction in {
            "v6_reference": v6_oof,
            "pair_v6_8_k1_q52_raw": pair_oof,
            "v7_blend85_15_round2": blend,
        }.items():
            oof_rows.append(
                pd.DataFrame(
                    {
                        "stage": args.stage,
                        "split_seed": seed,
                        "row": np.arange(len(train)),
                        "candidate": candidate,
                        "target": target,
                        "prediction": prediction,
                    }
                )
            )

    detail = pd.DataFrame(detail_rows)
    mean_row_delta = np.mean(np.column_stack(row_deltas), axis=1)
    ci_lower, ci_upper = cluster_interval(mean_row_delta, groups)
    summary = pd.DataFrame(
        [
            {
                "stage": args.stage,
                "seeds": ",".join(map(str, seeds)),
                "v6_mean_mae": detail["v6_mae"].mean(),
                "pair_mean_mae": detail["pair_mae"].mean(),
                "blend_mean_mae": detail["blend_mae"].mean(),
                "mean_delta_vs_v6": detail["delta_vs_v6"].mean(),
                "seed_win_rate": float((detail["delta_vs_v6"] < 0).mean()),
                "worst_seed_delta": detail["delta_vs_v6"].max(),
                "paired_ci95_lower": ci_lower,
                "paired_ci95_upper": ci_upper,
            }
        ]
    )
    prefix = f"v7_pair_blend_{args.stage}"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(
        args.results_dir / f"{prefix}_oof.csv", index=False
    )
    print(detail.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
