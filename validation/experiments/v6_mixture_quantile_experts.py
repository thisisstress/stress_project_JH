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
    paired_bootstrap_interval,
    validate_splits,
)


MODEL_SEED = 42
QUANTILES = [0.49, 0.50, 0.51, 0.52, 0.53]

EXPERT_WEIGHTS = {
    "balanced": {feature: 2 for feature in V6_FEATURES},
    "work_context": {
        "mean_working": 5,
        "missing_count": 3,
        "edu_level": 3,
        "medical_history": 1,
        "family_medical_history": 1,
        "sleep_pattern": 2,
        "age": 1,
    },
    "cardiometabolic": {
        "cholesterol": 3,
        "glucose": 3,
        "cholesterol_glucose_ratio": 3,
        "systolic_blood_pressure": 2,
        "diastolic_blood_pressure": 2,
        "pulse_pressure": 2,
        "mean_arterial_pressure": 2,
        "medical_history": 2,
        "family_medical_history": 2,
        "age": 1,
    },
    "body_composition": {
        "bmi": 4,
        "weight": 3,
        "height": 3,
        "bone_density": 4,
        "age": 2,
        "gender": 2,
    },
    "lifestyle": {
        "activity": 4,
        "sleep_pattern": 4,
        "smoke_status": 3,
        "gender": 1,
        "mean_working": 3,
        "edu_level": 2,
    },
    "missing_context": {
        "missing_count": 5,
        "mean_working": 2,
        "medical_history": 3,
        "family_medical_history": 3,
        "edu_level": 3,
        "sleep_pattern": 1,
    },
}

EXPERT_GROUPS = {
    "all_experts": tuple(EXPERT_WEIGHTS),
    "balanced_work_missing": (
        "balanced",
        "work_context",
        "missing_context",
    ),
    "balanced_body_cardio": (
        "balanced",
        "body_composition",
        "cardiometabolic",
    ),
    "balanced_lifestyle_work_missing": (
        "balanced",
        "lifestyle",
        "work_context",
        "missing_context",
    ),
    "work_lifestyle_missing": (
        "work_context",
        "lifestyle",
        "missing_context",
    ),
    "balanced_work_cardio_body": (
        "balanced",
        "work_context",
        "cardiometabolic",
        "body_composition",
    ),
    "balanced_work_cardio_missing": (
        "balanced",
        "work_context",
        "cardiometabolic",
        "missing_context",
    ),
}


def build_expert_features(
    raw_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    base = add_base_features(raw_features)
    return {
        name: apply_feature_weights(base, weights)
        for name, weights in EXPERT_WEIGHTS.items()
    }


def fit_expert_oof(
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


def build_candidates(
    expert_tree_predictions: dict[str, np.ndarray],
    trees_per_expert: int,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for expert_name, tree_predictions in expert_tree_predictions.items():
        candidate_trees = tree_predictions[:, :trees_per_expert]
        for quantile in QUANTILES:
            candidates[
                f"expert_{expert_name}_q{int(quantile * 100):02d}"
            ] = np.quantile(candidate_trees, quantile, axis=1)

    for group_name, expert_names in EXPERT_GROUPS.items():
        pooled = np.column_stack(
            [
                expert_tree_predictions[name][:, :trees_per_expert]
                for name in expert_names
            ]
        )
        for quantile in QUANTILES:
            candidates[
                f"pool_{group_name}_q{int(quantile * 100):02d}"
            ] = np.quantile(pooled, quantile, axis=1)

    q51_by_expert = np.column_stack(
        [
            np.quantile(tree_predictions, 0.51, axis=1)
            for tree_predictions in (
                item[:, :trees_per_expert]
                for item in expert_tree_predictions.values()
            )
        ]
    )
    candidates["aggregate_expert_mean_q51"] = q51_by_expert.mean(axis=1)
    candidates["aggregate_expert_median_q51"] = np.median(
        q51_by_expert, axis=1
    )
    candidates["v1_reference"] = np.quantile(
        expert_tree_predictions["balanced"], 0.51, axis=1
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
        choices=["screen", "development", "audit"],
        default="screen",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--reference-trees", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_trees < args.n_estimators:
        raise ValueError("Reference 트리 수는 전문가별 트리 수 이상이어야 합니다.")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].reset_index(drop=True)
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    expert_features = build_expert_features(raw_features)
    stage_seeds = {
        "screen": [DEVELOPMENT_SEEDS[0]],
        "development": DEVELOPMENT_SEEDS,
        "audit": AUDIT_SEEDS,
    }

    detail_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    for split_seed in stage_seeds[args.stage]:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=split_seed,
        )
        splits = list(splitter.split(raw_features, target_bins, groups))
        validate_splits(splits, groups, len(raw_features))
        expert_predictions = {}
        for expert_name, features in expert_features.items():
            expert_trees = (
                args.reference_trees
                if expert_name == "balanced"
                else args.n_estimators
            )
            expert_predictions[expert_name] = fit_expert_oof(
                features, target, splits, expert_trees
            )
            print(
                f"stage={args.stage} seed={split_seed} "
                f"expert={expert_name} complete",
                flush=True,
            )

        candidates = build_candidates(
            expert_predictions, trees_per_expert=args.n_estimators
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
                    "n_estimators_per_expert": args.n_estimators,
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
    prefix = f"v6_experts_{args.stage}_{args.n_estimators}trees"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    oof.to_csv(args.results_dir / f"{prefix}_oof.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
