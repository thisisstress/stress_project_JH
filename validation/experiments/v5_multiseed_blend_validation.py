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
    AUDIT_SEEDS,
    DEVELOPMENT_SEEDS,
    ID_COL,
    N_SPLITS,
    TARGET,
    V6_FEATURES,
    add_base_features,
    apply_feature_weights,
    make_duplicate_groups,
    make_preprocessor,
    make_target_bins,
    validate_splits,
)
from v5_interaction_blend_validation import summarize_candidates


BASE_SEED = 42
ADDITIONAL_SEEDS = [2026, 3407]
BASE_QUANTILE = 0.51
BLEND_WEIGHTS = [0.025, 0.05, 0.10, 0.25, 0.50, 1.00]


def fit_oof_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    base_trees: int,
    trees_per_aux_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base_prediction = np.zeros(len(features), dtype=float)
    auxiliary_predictions = {
        "pooled_q50_raw": np.zeros(len(features), dtype=float),
        "pooled_q50_round2": np.zeros(len(features), dtype=float),
        "pooled_q51_raw": np.zeros(len(features), dtype=float),
        "pooled_q51_round2": np.zeros(len(features), dtype=float),
        "seed_mean_q50_raw": np.zeros(len(features), dtype=float),
        "seed_mean_q51_raw": np.zeros(len(features), dtype=float),
        "seed_median_q50_raw": np.zeros(len(features), dtype=float),
        "seed_median_q51_raw": np.zeros(len(features), dtype=float),
    }

    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        preprocessor = make_preprocessor(features)
        train_matrix = preprocessor.fit_transform(features.iloc[train_index])
        valid_matrix = preprocessor.transform(features.iloc[valid_index])

        base_model = ExtraTreesRegressor(
            n_estimators=base_trees,
            min_samples_leaf=1,
            max_features=1,
            random_state=BASE_SEED,
            n_jobs=1,
        )
        base_model.fit(train_matrix, target.iloc[train_index])
        base_trees_prediction = np.column_stack(
            [
                tree.predict(valid_matrix)
                for tree in base_model.estimators_
            ]
        )
        base_prediction[valid_index] = np.quantile(
            base_trees_prediction, BASE_QUANTILE, axis=1
        )

        seed_tree_predictions = [
            base_trees_prediction[:, :trees_per_aux_seed]
        ]
        for model_seed in ADDITIONAL_SEEDS:
            model = ExtraTreesRegressor(
                n_estimators=trees_per_aux_seed,
                min_samples_leaf=1,
                max_features=1,
                random_state=model_seed,
                n_jobs=1,
            )
            model.fit(train_matrix, target.iloc[train_index])
            seed_tree_predictions.append(
                np.column_stack(
                    [
                        tree.predict(valid_matrix)
                        for tree in model.estimators_
                    ]
                )
            )

        pooled = np.column_stack(seed_tree_predictions)
        seed_q50 = np.column_stack(
            [np.quantile(block, 0.50, axis=1) for block in seed_tree_predictions]
        )
        seed_q51 = np.column_stack(
            [np.quantile(block, 0.51, axis=1) for block in seed_tree_predictions]
        )
        pooled_q50 = np.quantile(pooled, 0.50, axis=1)
        pooled_q51 = np.quantile(pooled, 0.51, axis=1)
        fold_predictions = {
            "pooled_q50_raw": pooled_q50,
            "pooled_q50_round2": np.round(pooled_q50, 2),
            "pooled_q51_raw": pooled_q51,
            "pooled_q51_round2": np.round(pooled_q51, 2),
            "seed_mean_q50_raw": seed_q50.mean(axis=1),
            "seed_mean_q51_raw": seed_q51.mean(axis=1),
            "seed_median_q50_raw": np.median(seed_q50, axis=1),
            "seed_median_q51_raw": np.median(seed_q51, axis=1),
        }
        for name, prediction in fold_predictions.items():
            auxiliary_predictions[name][valid_index] = prediction
        print(f"fold={fold}/{len(splits)} complete", flush=True)

    return base_prediction, auxiliary_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--stage",
        choices=["development", "audit"],
        default="development",
    )
    parser.add_argument("--base-trees", type=int, default=1200)
    parser.add_argument("--aux-trees-per-seed", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aux_trees_per_seed > args.base_trees:
        raise ValueError("Seed 42 보조 트리 수는 기본 트리 수 이하여야 합니다.")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].reset_index(drop=True)
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    features = apply_feature_weights(
        add_base_features(raw_features),
        {feature: 2 for feature in V6_FEATURES},
    )
    split_seeds = (
        DEVELOPMENT_SEEDS if args.stage == "development" else AUDIT_SEEDS
    )

    detail_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    for split_seed in split_seeds:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=split_seed,
        )
        splits = list(splitter.split(raw_features, target_bins, groups))
        validate_splits(splits, groups, len(raw_features))
        base_prediction, auxiliary_predictions = fit_oof_candidates(
            features,
            target,
            splits,
            args.base_trees,
            args.aux_trees_per_seed,
        )
        base_mae = mean_absolute_error(target, base_prediction)
        candidates = {"base_v1": base_prediction}
        for auxiliary_name, auxiliary_prediction in auxiliary_predictions.items():
            for blend_weight in BLEND_WEIGHTS:
                name = f"{auxiliary_name}_blend_{int(blend_weight * 1000):04d}"
                candidates[name] = np.clip(
                    (1.0 - blend_weight) * base_prediction
                    + blend_weight * auxiliary_prediction,
                    0.0,
                    1.0,
                )

        for candidate_name, prediction in candidates.items():
            if candidate_name == "base_v1":
                auxiliary_name = "none"
                blend_weight = 0.0
            else:
                auxiliary_name, weight_text = candidate_name.rsplit(
                    "_blend_", maxsplit=1
                )
                blend_weight = int(weight_text) / 1000.0
            mae = mean_absolute_error(target, prediction)
            detail_rows.append(
                {
                    "stage": args.stage,
                    "split_seed": split_seed,
                    "candidate": candidate_name,
                    "auxiliary": auxiliary_name,
                    "blend_weight": blend_weight,
                    "mae": mae,
                    "delta_vs_base": mae - base_mae,
                }
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        "stage": args.stage,
                        "split_seed": split_seed,
                        "row": np.arange(len(target)),
                        "candidate": candidate_name,
                        "target": target,
                        "prediction": prediction,
                    }
                )
            )
        print(
            f"stage={args.stage} split_seed={split_seed} complete",
            flush=True,
        )

    detail = pd.DataFrame(detail_rows)
    oof = pd.concat(oof_frames, ignore_index=True)
    summary = summarize_candidates(detail, oof, groups)
    prefix = f"v5_multiseed_blend_{args.stage}"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    oof.to_csv(args.results_dir / f"{prefix}_oof.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
