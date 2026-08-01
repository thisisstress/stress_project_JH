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
    paired_bootstrap_interval,
    validate_splits,
)


BASE_MODEL_SEED = 42
AUXILIARY_MODEL_SEEDS = [7, 2026, 7777]
BASE_QUANTILE = 0.51
AUXILIARY_QUANTILES = [0.50, 0.51]
BLEND_WEIGHTS = [0.025, 0.05, 0.075, 0.10]


def combine_categories(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    normalized = frame[columns].astype("string").fillna("__MISSING__")
    return normalized.agg("|".join, axis=1)


def add_interaction_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    """Test 전체 통계 없이 한 행 내부에서만 생성하는 상호작용."""
    result = add_base_features(raw_features)

    result["work_missing"] = result["mean_working"].isna().astype("int8")
    result["medical_missing"] = (
        result["medical_history"].isna().astype("int8")
    )
    result["family_missing"] = (
        result["family_medical_history"].isna().astype("int8")
    )
    result["education_missing"] = result["edu_level"].isna().astype("int8")
    result["history_missing_count"] = (
        result["medical_missing"] + result["family_missing"]
    )

    result["history_pair"] = combine_categories(
        result, ["medical_history", "family_medical_history"]
    )
    result["sleep_activity"] = combine_categories(
        result, ["sleep_pattern", "activity"]
    )
    result["smoke_activity"] = combine_categories(
        result, ["smoke_status", "activity"]
    )
    result["education_sleep"] = combine_categories(
        result, ["edu_level", "sleep_pattern"]
    )
    result["gender_smoke"] = combine_categories(
        result, ["gender", "smoke_status"]
    )
    result["history_sleep"] = combine_categories(
        result, ["medical_history", "sleep_pattern"]
    )
    result["education_work_missing"] = combine_categories(
        result, ["edu_level", "work_missing"]
    )

    result["age_working"] = result["age"] * result["mean_working"]
    result["age_bmi"] = result["age"] * result["bmi"]
    result["age_bone_density"] = result["age"] * result["bone_density"]
    result["metabolic_product"] = (
        result["cholesterol"] * result["glucose"]
    )
    result["age_mean_arterial_pressure"] = (
        result["age"] * result["mean_arterial_pressure"]
    )
    return result


def build_feature_sets(
    raw_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = add_base_features(raw_features)
    base_features = apply_feature_weights(
        base, {feature: 2 for feature in V6_FEATURES}
    )

    interactions = add_interaction_features(raw_features)
    auxiliary_weights = {
        **{feature: 2 for feature in V6_FEATURES},
        "bone_density": 3,
        "missing_count": 2,
        "history_missing_count": 2,
        "history_pair": 2,
        "sleep_activity": 1,
        "education_sleep": 1,
        "history_sleep": 1,
        "education_work_missing": 2,
        "age_working": 1,
        "age_bmi": 1,
    }
    auxiliary_features = apply_feature_weights(
        interactions, auxiliary_weights
    )
    return base_features, auxiliary_features


def fit_quantile_oof(
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_seeds: list[int],
    n_estimators: int,
    quantiles: list[float],
) -> dict[tuple[int, float], np.ndarray]:
    predictions = {
        (model_seed, quantile): np.zeros(len(features), dtype=float)
        for model_seed in model_seeds
        for quantile in quantiles
    }

    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        preprocessor = make_preprocessor(features)
        train_matrix = preprocessor.fit_transform(features.iloc[train_index])
        valid_matrix = preprocessor.transform(features.iloc[valid_index])

        for model_seed in model_seeds:
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=1,
                max_features=1,
                random_state=model_seed,
                n_jobs=1,
            )
            model.fit(train_matrix, target.iloc[train_index])
            tree_predictions = np.column_stack(
                [tree.predict(valid_matrix) for tree in model.estimators_]
            )
            for quantile in quantiles:
                predictions[(model_seed, quantile)][valid_index] = (
                    np.quantile(tree_predictions, quantile, axis=1)
                )

        print(
            f"fold={fold}/{len(splits)} seeds={model_seeds} complete",
            flush=True,
        )
    return predictions


def auxiliary_candidates(
    seed_predictions: dict[tuple[int, float], np.ndarray],
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for quantile in AUXILIARY_QUANTILES:
        stacked = np.column_stack(
            [
                seed_predictions[(model_seed, quantile)]
                for model_seed in AUXILIARY_MODEL_SEEDS
            ]
        )
        quantile_name = f"q{int(round(quantile * 100)):02d}"
        for column, model_seed in enumerate(AUXILIARY_MODEL_SEEDS):
            candidates[f"interaction_seed_{model_seed}_{quantile_name}"] = (
                stacked[:, column]
            )
        candidates[f"interaction_seed_mean_{quantile_name}"] = stacked.mean(
            axis=1
        )
        candidates[f"interaction_seed_median_{quantile_name}"] = np.median(
            stacked, axis=1
        )
    return candidates


def summarize_candidates(
    detail: pd.DataFrame,
    oof: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    rows = []
    baseline = (
        oof[oof["candidate"] == "base_v1"]
        .sort_values(["split_seed", "row"])
        .reset_index(drop=True)
    )
    baseline_loss = np.abs(
        baseline["target"].to_numpy() - baseline["prediction"].to_numpy()
    )

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
                "auxiliary": candidate_detail["auxiliary"].iloc[0],
                "blend_weight": candidate_detail["blend_weight"].iloc[0],
                "mean_mae": candidate_detail["mae"].mean(),
                "seed_std": candidate_detail["mae"].std(ddof=0),
                "worst_seed_mae": candidate_detail["mae"].max(),
                "best_seed_mae": candidate_detail["mae"].min(),
                "mean_delta_vs_base": candidate_detail[
                    "delta_vs_base"
                ].mean(),
                "seed_win_rate_vs_base": (
                    candidate_detail["delta_vs_base"] < 0
                ).mean(),
                "sample_win_rate_vs_base": (
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
        choices=["development", "audit"],
        default="development",
    )
    parser.add_argument("--base-trees", type=int, default=1200)
    parser.add_argument("--aux-trees", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].reset_index(drop=True)
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    base_features, auxiliary_features = build_feature_sets(raw_features)
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
        splits = list(
            splitter.split(raw_features, target_bins, groups)
        )
        validate_splits(splits, groups, len(raw_features))

        base_prediction = fit_quantile_oof(
            base_features,
            target,
            splits,
            [BASE_MODEL_SEED],
            args.base_trees,
            [BASE_QUANTILE],
        )[(BASE_MODEL_SEED, BASE_QUANTILE)]
        base_mae = mean_absolute_error(target, base_prediction)
        auxiliary_seed_predictions = fit_quantile_oof(
            auxiliary_features,
            target,
            splits,
            AUXILIARY_MODEL_SEEDS,
            args.aux_trees,
            AUXILIARY_QUANTILES,
        )
        candidates = {"base_v1": base_prediction}
        for auxiliary_name, auxiliary_prediction in auxiliary_candidates(
            auxiliary_seed_predictions
        ).items():
            for blend_weight in BLEND_WEIGHTS:
                candidate_name = (
                    f"{auxiliary_name}_blend_{int(blend_weight * 1000):03d}"
                )
                candidates[candidate_name] = np.clip(
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
    prefix = f"v5_interaction_blend_{args.stage}"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    oof.to_csv(args.results_dir / f"{prefix}_oof.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
