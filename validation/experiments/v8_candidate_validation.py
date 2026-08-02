# -*- coding: utf-8 -*-
"""Validate fixed V8 candidates against the exact current-code V7 reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in [VALIDATION_DIR, EXPERIMENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
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
from v8_confidence_pair_screen import (
    DEFAULT_TREES,
    FEATURE_COPY_COUNTS,
    neighbor_bundle,
    tree_bundle,
)


CONFIRM_SEEDS = [2026, 3407]
AUDIT4_SEEDS = [324161, 350377, 376573]
BOOTSTRAP_SEED = 20260804


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
        sampled = rng.integers(0, count, size=count)
        estimates[draw] = sums[sampled].sum() / sizes[sampled].sum()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def fixed_predictions(
    v6: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pair = auxiliary["pair_all_q52"]
    triple = auxiliary["triple_closest30_q52"]
    pooled = auxiliary["pair_triple_pool_q52"]
    triple_std = meta["triple_std"]
    pair_distance = meta["pair_distance_mean"]

    triple_gate_weight = np.where(
        triple_std <= 0.21804926296075186,
        0.25,
        np.where(triple_std <= 0.2776087820529938, 0.30, 0.10),
    )
    distance_gate_weight = np.where(
        pair_distance <= 0.0035416720590243735,
        0.30,
        np.where(pair_distance <= 0.01013889101644357, 0.15, 0.30),
    )

    recipes = {
        "v6_reference": v6,
        "v7_reference": 0.85 * v6 + 0.15 * pair,
        "v8_triple_gate": (
            (1.0 - triple_gate_weight) * v6
            + triple_gate_weight * pair
        ),
        "v8_pair_distance_gate": (
            (1.0 - distance_gate_weight) * v6
            + distance_gate_weight * pair
        ),
        "v8_triple_scalar_w275": 0.725 * v6 + 0.275 * triple,
        "v8_pair_triple_pool_w275": 0.725 * v6 + 0.275 * pooled,
    }
    return {
        name: np.clip(np.round(prediction, 2), 0.0, 1.0)
        for name, prediction in recipes.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["confirm", "audit4"], required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_TREES)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or (
        CONFIRM_SEEDS if args.stage == "confirm" else AUDIT4_SEEDS
    )

    train = pd.read_csv(args.data_dir / "train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].astype(float).reset_index(drop=True)
    base = add_base_features(raw)
    weighted = apply_feature_weights(base, FEATURE_COPY_COUNTS)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(target)

    detail_rows = []
    row_deltas: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(raw, bins, groups))
        validate_splits(splits, groups, len(train))
        prediction_oof: dict[str, np.ndarray] = {}

        for fold, (train_index, valid_index) in enumerate(splits):
            v6, tree_meta = tree_bundle(
                weighted.iloc[train_index],
                weighted.iloc[valid_index],
                target.iloc[train_index],
                args.n_estimators,
            )
            auxiliary, neighbor_meta = neighbor_bundle(
                base.iloc[train_index],
                base.iloc[valid_index],
                target.iloc[train_index].to_numpy(),
            )
            fold_predictions = fixed_predictions(
                v6,
                auxiliary,
                {**tree_meta, **neighbor_meta},
            )
            for name, values in fold_predictions.items():
                if name not in prediction_oof:
                    prediction_oof[name] = np.full(len(train), np.nan)
                prediction_oof[name][valid_index] = values
            print(
                f"stage={args.stage} seed={seed} fold={fold} complete",
                flush=True,
            )

        reference_error = np.abs(
            target.to_numpy() - prediction_oof["v7_reference"]
        )
        for name, prediction in prediction_oof.items():
            error = np.abs(target.to_numpy() - prediction)
            delta = error - reference_error
            detail_rows.append(
                {
                    "stage": args.stage,
                    "split_seed": seed,
                    "candidate": name,
                    "mae": error.mean(),
                    "delta_vs_v7": delta.mean(),
                }
            )
            row_deltas.setdefault(name, []).append(delta)

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for name, group in detail.groupby("candidate"):
        mean_row_delta = np.mean(np.column_stack(row_deltas[name]), axis=1)
        ci_lower, ci_upper = cluster_interval(mean_row_delta, groups)
        summary_rows.append(
            {
                "stage": args.stage,
                "candidate": name,
                "mean_mae": group["mae"].mean(),
                "mean_delta_vs_v7": group["delta_vs_v7"].mean(),
                "seed_win_rate": float((group["delta_vs_v7"] < 0).mean()),
                "worst_seed_delta": group["delta_vs_v7"].max(),
                "paired_ci95_lower": ci_lower,
                "paired_ci95_upper": ci_upper,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mae")
    prefix = f"v8_candidate_{args.stage}"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    print(detail.sort_values(["split_seed", "mae"]).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
