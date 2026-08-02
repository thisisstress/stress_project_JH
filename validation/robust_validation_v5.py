from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TARGET = "stress_score"
ID_COL = "ID"

V6_FEATURES = [
    "mean_working",
    "bmi",
    "cholesterol",
    "height",
    "glucose",
    "weight",
    "cholesterol_glucose_ratio",
    "bone_density",
]


DEVELOPMENT_SEEDS = [42, 2026, 3407]
# 2026-08-01 이전 실험에서 사용하지 않은 고정 감사 Seed입니다.
AUDIT_SEEDS = [104729, 130363, 155921]
N_SPLITS = 5
N_TARGET_BINS = 10
MODEL_SEED = 42
BOOTSTRAP_SEED = 20260801


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    configuration: str


MODEL_SPECS = [
    ModelSpecification("v1", "base q=0.51 raw"),
    ModelSpecification("v2", "work_bone q=0.50 round(2)"),
    ModelSpecification("v3", "0.65 work_bone + 0.35 stable"),
    ModelSpecification("v4", "base q=0.505 raw"),
]


def add_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    """다른 행이나 데이터셋 통계를 사용하지 않는 행 단위 파생변수."""
    result = frame.copy()
    height_m = result["height"] / 100.0
    sbp = result["systolic_blood_pressure"]
    dbp = result["diastolic_blood_pressure"]
    result["bmi"] = result["weight"] / height_m.pow(2)
    result["pulse_pressure"] = sbp - dbp
    result["mean_arterial_pressure"] = (sbp + 2.0 * dbp) / 3.0
    result["blood_pressure_ratio"] = sbp / (dbp + 1e-6)
    result["cholesterol_glucose_ratio"] = result["cholesterol"] / (
        result["glucose"] + 1e-6
    )
    result["missing_count"] = result.isna().sum(axis=1)
    return result


def apply_feature_weights(
    frame: pd.DataFrame,
    weights: dict[str, int],
) -> pd.DataFrame:
    result = frame.copy()
    for feature, additional_copies in weights.items():
        for copy_index in range(1, additional_copies + 1):
            result[f"{feature}__copy{copy_index}"] = result[feature]
    return result


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerical = frame.select_dtypes(include=["number"]).columns.tolist()
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="missing"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numerical",
                SimpleImputer(strategy="median", add_indicator=True),
                numerical,
            ),
        ]
    )


def make_duplicate_groups(raw_features: pd.DataFrame) -> pd.Series:
    normalized = raw_features.astype(object).where(
        raw_features.notna(), "__MISSING__"
    )
    return pd.util.hash_pandas_object(normalized, index=False)


def make_target_bins(target: pd.Series) -> pd.Series:
    return pd.qcut(
        target.rank(method="first"),
        q=N_TARGET_BINS,
        labels=False,
    )


def build_feature_sets(raw_features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base_features = add_base_features(raw_features)
    base_weights = {feature: 2 for feature in V6_FEATURES}

    work_bone_weights = dict(base_weights)
    work_bone_weights["mean_working"] = 1
    work_bone_weights["bone_density"] = 4

    stable_weights = dict(base_weights)
    stable_weights["glucose"] = 3
    stable_weights["pulse_pressure"] = 1
    stable_weights["bone_density"] = 4

    return {
        "base": apply_feature_weights(base_features, base_weights),
        "work_bone": apply_feature_weights(
            base_features, work_bone_weights
        ),
        "stable": apply_feature_weights(base_features, stable_weights),
    }


def validate_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: pd.Series,
    row_count: int,
) -> None:
    validation_count = np.zeros(row_count, dtype=np.int8)
    group_values = groups.to_numpy()
    for train_index, valid_index in splits:
        validation_count[valid_index] += 1
        train_groups = set(group_values[train_index])
        valid_groups = set(group_values[valid_index])
        if train_groups.intersection(valid_groups):
            raise RuntimeError("동일 입력 그룹이 Train과 Validation에 나뉘었습니다.")
    if not np.all(validation_count == 1):
        raise RuntimeError("각 행은 정확히 한 Validation Fold에 포함되어야 합니다.")


def fit_tree_oof(
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree_oof = np.zeros(
        (len(features), n_estimators), dtype=np.float32
    )
    fold_by_row = np.full(len(features), -1, dtype=np.int8)

    for fold, (train_index, valid_index) in enumerate(splits):
        preprocessor = make_preprocessor(features)
        train_matrix = preprocessor.fit_transform(
            features.iloc[train_index]
        )
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
        fold_by_row[valid_index] = fold

    return tree_oof, fold_by_row


def model_predictions(
    tree_predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    base = tree_predictions["base"]
    work_bone = tree_predictions["work_bone"]
    stable = tree_predictions["stable"]

    v1 = np.quantile(base, 0.51, axis=1)
    v2 = np.round(np.quantile(work_bone, 0.50, axis=1), 2)
    work_bone_raw = np.quantile(work_bone, 0.50, axis=1)
    stable_raw = np.quantile(stable, 0.50, axis=1)
    v3 = 0.65 * work_bone_raw + 0.35 * stable_raw
    v4 = np.quantile(base, 0.505, axis=1)
    return {"v1": v1, "v2": v2, "v3": v3, "v4": v4}


def fold_balance_rows(
    target: pd.Series,
    target_bins: pd.Series,
    fold_by_row: np.ndarray,
    stage: str,
    split_seed: int,
) -> list[dict]:
    rows = []
    for fold in range(N_SPLITS):
        index = np.flatnonzero(fold_by_row == fold)
        bin_counts = target_bins.iloc[index].value_counts().sort_index()
        rows.append(
            {
                "stage": stage,
                "split_seed": split_seed,
                "fold": fold,
                "rows": len(index),
                "target_mean": float(target.iloc[index].mean()),
                "target_std": float(target.iloc[index].std()),
                "target_min": float(target.iloc[index].min()),
                "target_max": float(target.iloc[index].max()),
                "smallest_bin": int(bin_counts.min()),
                "largest_bin": int(bin_counts.max()),
            }
        )
    return rows


def paired_bootstrap_interval(
    loss_difference: np.ndarray,
    groups: pd.Series,
    iterations: int = 4000,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    group_codes, unique_groups = pd.factorize(groups, sort=True)
    group_sums = np.bincount(
        group_codes, weights=loss_difference, minlength=len(unique_groups)
    )
    group_counts = np.bincount(
        group_codes, minlength=len(unique_groups)
    )
    bootstrap_means = np.empty(iterations, dtype=float)

    batch_size = 250
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        sampled_codes = rng.integers(
            0,
            len(unique_groups),
            size=(stop - start, len(unique_groups)),
        )
        sampled_sums = group_sums[sampled_codes].sum(axis=1)
        sampled_counts = group_counts[sampled_codes].sum(axis=1)
        bootstrap_means[start:stop] = sampled_sums / sampled_counts

    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize(
    detail: pd.DataFrame,
    oof: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    rows = []
    for stage in detail["stage"].drop_duplicates():
        stage_detail = detail[detail["stage"] == stage]
        stage_oof = oof[oof["stage"] == stage]
        for specification in MODEL_SPECS:
            model_detail = stage_detail[
                stage_detail["model"] == specification.name
            ]
            candidate_rows = stage_oof[
                stage_oof["model"] == specification.name
            ].sort_values(["split_seed", "row"])
            baseline_rows = stage_oof[
                stage_oof["model"] == "v1"
            ].sort_values(["split_seed", "row"])

            candidate_loss = np.abs(
                candidate_rows["target"].to_numpy()
                - candidate_rows["prediction"].to_numpy()
            )
            baseline_loss = np.abs(
                baseline_rows["target"].to_numpy()
                - baseline_rows["prediction"].to_numpy()
            )
            seed_count = model_detail["split_seed"].nunique()
            row_loss_difference = (
                candidate_loss.reshape(seed_count, -1)
                - baseline_loss.reshape(seed_count, -1)
            ).mean(axis=0)
            ci_lower, ci_upper = paired_bootstrap_interval(
                row_loss_difference, groups
            )
            rows.append(
                {
                    "stage": stage,
                    "model": specification.name,
                    "configuration": specification.configuration,
                    "seeds": seed_count,
                    "mean_mae": float(model_detail["mae"].mean()),
                    "seed_std": float(model_detail["mae"].std(ddof=0)),
                    "worst_seed_mae": float(model_detail["mae"].max()),
                    "best_seed_mae": float(model_detail["mae"].min()),
                    "mean_delta_vs_v1": float(
                        model_detail["delta_vs_v1"].mean()
                    ),
                    "seed_win_rate_vs_v1": float(
                        (model_detail["delta_vs_v1"] < 0).mean()
                    ),
                    "sample_win_rate_vs_v1": float(
                        (row_loss_difference < 0).mean()
                    ),
                    "paired_delta_ci95_lower": ci_lower,
                    "paired_delta_ci95_upper": ci_upper,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["stage", "mean_mae", "worst_seed_mae"]
    )


def data_profile(
    raw_features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> dict:
    duplicated = raw_features.duplicated(keep=False)
    duplicate_frame = pd.DataFrame(
        {"group": groups[duplicated], "target": target[duplicated]}
    )
    conflicting_groups = int(
        (duplicate_frame.groupby("group")["target"].nunique() > 1).sum()
    )
    return {
        "rows": len(raw_features),
        "input_columns": raw_features.shape[1],
        "target_unique_values": int(target.nunique()),
        "duplicate_affected_rows": int(duplicated.sum()),
        "duplicate_groups": int(groups[duplicated].nunique()),
        "duplicate_groups_with_conflicting_target": conflicting_groups,
        "missing_cells": int(raw_features.isna().sum().sum()),
        "development_seeds": DEVELOPMENT_SEEDS,
        "audit_seeds": AUDIT_SEEDS,
        "audit_seed_status": "first_consumed_on_2026-08-01",
        "splitter": (
            "StratifiedGroupKFold(5, target quantile bins=10, "
            "exact-input duplicate groups)"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["development", "audit"],
        default=["development", "audit"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[ID_COL, TARGET])
    target = train[TARGET]
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    feature_sets = build_feature_sets(raw_features)

    stage_seeds = {
        "development": DEVELOPMENT_SEEDS,
        "audit": AUDIT_SEEDS,
    }
    detail_rows: list[dict] = []
    balance_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []

    for stage in args.stages:
        for split_seed in stage_seeds[stage]:
            splitter = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=split_seed,
            )
            splits = list(
                splitter.split(raw_features, target_bins, groups)
            )
            validate_splits(splits, groups, len(raw_features))
            tree_predictions: dict[str, np.ndarray] = {}
            fold_by_row = None

            for configuration, features in feature_sets.items():
                tree_oof, configuration_folds = fit_tree_oof(
                    features,
                    target,
                    splits,
                    n_estimators=args.n_estimators,
                )
                tree_predictions[configuration] = tree_oof
                if fold_by_row is None:
                    fold_by_row = configuration_folds
                elif not np.array_equal(
                    fold_by_row, configuration_folds
                ):
                    raise RuntimeError("모델별 Fold가 일치하지 않습니다.")
                print(
                    f"stage={stage} seed={split_seed} "
                    f"config={configuration} complete",
                    flush=True,
                )

            assert fold_by_row is not None
            predictions = model_predictions(tree_predictions)
            baseline_mae = mean_absolute_error(target, predictions["v1"])
            balance_rows.extend(
                fold_balance_rows(
                    target,
                    target_bins,
                    fold_by_row,
                    stage,
                    split_seed,
                )
            )

            for specification in MODEL_SPECS:
                prediction = predictions[specification.name]
                fold_scores = [
                    mean_absolute_error(
                        target.iloc[np.flatnonzero(fold_by_row == fold)],
                        prediction[fold_by_row == fold],
                    )
                    for fold in range(N_SPLITS)
                ]
                mae = float(mean_absolute_error(target, prediction))
                detail_rows.append(
                    {
                        "stage": stage,
                        "split_seed": split_seed,
                        "model": specification.name,
                        "configuration": specification.configuration,
                        "mae": mae,
                        "delta_vs_v1": mae - baseline_mae,
                        "fold_mean": float(np.mean(fold_scores)),
                        "fold_std": float(np.std(fold_scores)),
                        "worst_fold": float(np.max(fold_scores)),
                        "best_fold": float(np.min(fold_scores)),
                    }
                )
                oof_frames.append(
                    pd.DataFrame(
                        {
                            "stage": stage,
                            "split_seed": split_seed,
                            "row": np.arange(len(target)),
                            "duplicate_group": groups.astype(str),
                            "fold": fold_by_row,
                            "model": specification.name,
                            "target": target,
                            "prediction": prediction,
                        }
                    )
                )

            del tree_predictions

    detail = pd.DataFrame(detail_rows)
    oof = pd.concat(oof_frames, ignore_index=True)
    summary = summarize(detail, oof, groups)
    balance = pd.DataFrame(balance_rows)
    profile = data_profile(raw_features, target, groups)

    detail.to_csv(
        args.results_dir / "robust_validation_v5_detail.csv", index=False
    )
    summary.to_csv(
        args.results_dir / "robust_validation_v5_summary.csv", index=False
    )
    oof.to_csv(
        args.results_dir / "robust_validation_v5_oof.csv", index=False
    )
    balance.to_csv(
        args.results_dir / "robust_validation_v5_fold_balance.csv",
        index=False,
    )
    with (args.results_dir / "robust_validation_v5_profile.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(profile, file, ensure_ascii=False, indent=2)

    print("\nRobust validation summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
