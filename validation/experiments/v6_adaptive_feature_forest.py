from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

from robust_validation_v5 import (
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
    paired_bootstrap_interval,
    validate_splits,
)


MODEL_SEED = 42
AUDIT2_SEEDS = [181081, 196613, 214627]
QUANTILES = [0.495, 0.500, 0.505, 0.510, 0.515, 0.520, 0.525]

BASE_WEIGHTS = {feature: 2 for feature in V6_FEATURES}
PROFILE_WEIGHTS = {
    "base": BASE_WEIGHTS,
    "work1_bmi4": {
        **BASE_WEIGHTS,
        "mean_working": 1,
        "bmi": 4,
    },
    "work1_bmi4_bone4": {
        **BASE_WEIGHTS,
        "mean_working": 1,
        "bmi": 4,
        "bone_density": 4,
    },
    "ratio4_bone4": {
        **BASE_WEIGHTS,
        "cholesterol_glucose_ratio": 4,
        "bone_density": 4,
    },
    "glucose3_pulse1_bone4": {
        **BASE_WEIGHTS,
        "glucose": 3,
        "pulse_pressure": 1,
        "bone_density": 4,
    },
    "work1_bmi4_bone4_glucose3": {
        **BASE_WEIGHTS,
        "mean_working": 1,
        "bmi": 4,
        "bone_density": 4,
        "glucose": 3,
    },
    "work1_bmi4_bone4_ratio4": {
        **BASE_WEIGHTS,
        "mean_working": 1,
        "bmi": 4,
        "bone_density": 4,
        "cholesterol_glucose_ratio": 4,
    },
}


def fit_tree_oof(
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
) -> np.ndarray:
    tree_oof = np.zeros(
        (len(features), n_estimators), dtype=np.float32
    )
    for train_index, valid_index in splits:
        preprocessor = make_preprocessor(features)
        train_matrix = preprocessor.fit_transform(features.iloc[train_index])
        valid_matrix = preprocessor.transform(features.iloc[valid_index])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=1,
            max_features=1,
            random_state=MODEL_SEED,
            n_jobs=1,
        )
        model.fit(train_matrix, target.iloc[train_index])
        tree_oof[valid_index] = np.column_stack(
            [tree.predict(valid_matrix) for tree in model.estimators_]
        )
    return tree_oof


def candidate_predictions(
    profile_trees: dict[str, np.ndarray],
    fixed_profile: str | None = None,
    fixed_quantile: float | None = None,
    fixed_round2: bool = False,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    if fixed_profile is not None:
        if fixed_quantile is None:
            raise ValueError("고정 후보에는 fixed_quantile이 필요합니다.")
        raw = np.quantile(
            profile_trees[fixed_profile], fixed_quantile, axis=1
        )
        prediction = np.round(raw, 2) if fixed_round2 else raw
        suffix = "round2" if fixed_round2 else "raw"
        quantile_name = f"q{int(round(fixed_quantile * 1000)):03d}"
        candidates[
            f"{fixed_profile}_{quantile_name}_{suffix}"
        ] = prediction
        candidates["v1_reference"] = np.quantile(
            profile_trees["base"], 0.51, axis=1
        )
        return candidates

    for profile, tree_predictions in profile_trees.items():
        for quantile in QUANTILES:
            raw = np.quantile(tree_predictions, quantile, axis=1)
            quantile_name = f"q{int(round(quantile * 1000)):03d}"
            candidates[f"{profile}_{quantile_name}_raw"] = raw
            candidates[f"{profile}_{quantile_name}_round2"] = np.round(
                raw, 2
            )
    candidates["v1_reference"] = np.quantile(
        profile_trees["base"], 0.51, axis=1
    )
    return candidates


def summarize(
    detail: pd.DataFrame,
    oof: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    baseline = (
        oof[oof["candidate"] == "v1_reference"]
        .sort_values(["split_seed", "row"])
        .reset_index(drop=True)
    )
    baseline_loss = np.abs(
        baseline["target"].to_numpy() - baseline["prediction"].to_numpy()
    )
    rows = []
    for candidate in detail["candidate"].drop_duplicates():
        candidate_detail = detail[detail["candidate"] == candidate]
        candidate_oof = (
            oof[oof["candidate"] == candidate]
            .sort_values(["split_seed", "row"])
            .reset_index(drop=True)
        )
        candidate_loss = np.abs(
            candidate_oof["target"].to_numpy()
            - candidate_oof["prediction"].to_numpy()
        )
        seed_count = candidate_detail["split_seed"].nunique()
        row_loss_difference = (
            candidate_loss.reshape(seed_count, -1)
            - baseline_loss.reshape(seed_count, -1)
        ).mean(axis=0)
        ci_lower, ci_upper = paired_bootstrap_interval(
            row_loss_difference, groups
        )
        rows.append(
            {
                "candidate": candidate,
                "mean_mae": candidate_detail["mae"].mean(),
                "seed_std": candidate_detail["mae"].std(ddof=0),
                "worst_seed_mae": candidate_detail["mae"].max(),
                "best_seed_mae": candidate_detail["mae"].min(),
                "mean_delta_vs_v1": candidate_detail[
                    "delta_vs_v1"
                ].mean(),
                "seed_win_rate_vs_v1": (
                    candidate_detail["delta_vs_v1"] < 0
                ).mean(),
                "sample_win_rate_vs_v1": (
                    row_loss_difference < 0
                ).mean(),
                "paired_delta_ci95_lower": ci_lower,
                "paired_delta_ci95_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_mae", "worst_seed_mae", "paired_delta_ci95_upper"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--stage",
        choices=["development", "audit2"],
        default="development",
    )
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument(
        "--profile",
        nargs="+",
        choices=list(PROFILE_WEIGHTS),
    )
    parser.add_argument(
        "--fixed-profile",
        choices=list(PROFILE_WEIGHTS),
    )
    parser.add_argument("--fixed-quantile", type=float)
    parser.add_argument("--fixed-round2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].reset_index(drop=True)
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    base = add_base_features(raw_features)
    if args.fixed_profile is not None:
        selected_profiles = [args.fixed_profile]
    else:
        selected_profiles = args.profile or list(PROFILE_WEIGHTS)
    if "base" not in selected_profiles:
        selected_profiles = ["base", *selected_profiles]
    feature_sets = {
        profile: apply_feature_weights(base, PROFILE_WEIGHTS[profile])
        for profile in selected_profiles
    }
    split_seeds = (
        DEVELOPMENT_SEEDS if args.stage == "development" else AUDIT2_SEEDS
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
        profile_trees = {}
        for profile, features in feature_sets.items():
            profile_trees[profile] = fit_tree_oof(
                features, target, splits, args.n_estimators
            )
            print(
                f"stage={args.stage} seed={split_seed} "
                f"profile={profile} complete",
                flush=True,
            )

        candidates = candidate_predictions(
            profile_trees,
            fixed_profile=args.fixed_profile,
            fixed_quantile=args.fixed_quantile,
            fixed_round2=args.fixed_round2,
        )
        reference_mae = mean_absolute_error(
            target, candidates["v1_reference"]
        )
        for candidate, prediction in candidates.items():
            mae = mean_absolute_error(target, prediction)
            detail_rows.append(
                {
                    "stage": args.stage,
                    "split_seed": split_seed,
                    "candidate": candidate,
                    "n_estimators": args.n_estimators,
                    "mae": mae,
                    "delta_vs_v1": mae - reference_mae,
                }
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        "stage": args.stage,
                        "split_seed": split_seed,
                        "row": np.arange(len(target)),
                        "candidate": candidate,
                        "target": target,
                        "prediction": prediction,
                    }
                )
            )

    detail = pd.DataFrame(detail_rows)
    oof = pd.concat(oof_frames, ignore_index=True)
    summary = summarize(detail, oof, groups)
    prefix = f"v6_adaptive_{args.stage}_{args.n_estimators}trees"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    oof.to_csv(args.results_dir / f"{prefix}_oof.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
