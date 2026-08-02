# -*- coding: utf-8 -*-
"""Screen fold-local supervised interaction forests against cached V7 OOF."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import combinations
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
from sklearn.feature_selection import mutual_info_regression
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
SCREEN_SEED = 42
QUANTILES = [0.48, 0.50, 0.52, 0.54, 0.56]
BLEND_WEIGHTS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
NUMERIC_COLUMNS = [
    "age",
    "height",
    "weight",
    "cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "glucose",
    "bone_density",
    "mean_working",
    "bmi",
    "pulse_pressure",
    "mean_arterial_pressure",
    "blood_pressure_ratio",
    "cholesterol_glucose_ratio",
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


@dataclass(frozen=True)
class Profile:
    name: str
    top_count: int
    interaction_copies: int
    max_features: int | str


PROFILES = [
    Profile("int20_c2_mf1", 20, 2, 1),
    Profile("int50_c1_mf1", 50, 1, 1),
    Profile("int50_c2_mf2", 50, 2, 2),
    Profile("int100_c1_mf4", 100, 1, 4),
    Profile("int100_c0_sqrt", 100, 0, "sqrt"),
]


def normalized_numeric_pair(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_numeric = train_frame[NUMERIC_COLUMNS].astype(float)
    valid_numeric = valid_frame[NUMERIC_COLUMNS].astype(float)
    medians = train_numeric.median()
    filled_train = train_numeric.fillna(medians)
    filled_valid = valid_numeric.fillna(medians)
    scale = (filled_train.quantile(0.75) - filled_train.quantile(0.25)).replace(
        0.0, 1.0
    )
    train_z = ((filled_train - medians) / scale).clip(-10.0, 10.0)
    valid_z = ((filled_valid - medians) / scale).clip(-10.0, 10.0)

    train_values: dict[str, np.ndarray] = {}
    valid_values: dict[str, np.ndarray] = {}
    for first, second in combinations(NUMERIC_COLUMNS, 2):
        first_train = train_z[first].to_numpy()
        second_train = train_z[second].to_numpy()
        first_valid = valid_z[first].to_numpy()
        second_valid = valid_z[second].to_numpy()
        prefix = f"{first}__{second}"
        train_values[f"{prefix}__product"] = first_train * second_train
        valid_values[f"{prefix}__product"] = first_valid * second_valid
        train_values[f"{prefix}__difference"] = first_train - second_train
        valid_values[f"{prefix}__difference"] = first_valid - second_valid
        train_values[f"{prefix}__sum"] = first_train + second_train
        valid_values[f"{prefix}__sum"] = first_valid + second_valid
        train_values[f"{prefix}__ratio"] = first_train / (
            np.abs(second_train) + 0.5
        )
        valid_values[f"{prefix}__ratio"] = first_valid / (
            np.abs(second_valid) + 0.5
        )
    return pd.DataFrame(train_values), pd.DataFrame(valid_values)


def select_interactions(
    train_interactions: pd.DataFrame,
    target: pd.Series,
) -> list[str]:
    scores = mutual_info_regression(
        train_interactions,
        target,
        discrete_features=False,
        n_neighbors=5,
        random_state=MODEL_SEED,
        n_jobs=1,
    )
    order = np.argsort(scores)[::-1]
    return train_interactions.columns[order].tolist()


def build_profile_features(
    base_train: pd.DataFrame,
    base_valid: pd.DataFrame,
    interaction_train: pd.DataFrame,
    interaction_valid: pd.DataFrame,
    ranked_interactions: list[str],
    profile: Profile,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = ranked_interactions[: profile.top_count]
    train_features = apply_feature_weights(base_train, FEATURE_COPY_COUNTS)
    valid_features = apply_feature_weights(base_valid, FEATURE_COPY_COUNTS)
    for column in selected:
        train_features[f"interaction__{column}"] = interaction_train[column].to_numpy()
        valid_features[f"interaction__{column}"] = interaction_valid[column].to_numpy()
        for copy_index in range(1, profile.interaction_copies + 1):
            train_features[
                f"interaction__{column}__copy{copy_index}"
            ] = interaction_train[column].to_numpy()
            valid_features[
                f"interaction__{column}__copy{copy_index}"
            ] = interaction_valid[column].to_numpy()
    return train_features, valid_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v7-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()

    cache = np.load(args.v7_cache)
    cached_target = cache["target"].astype(float)
    cached_v6 = cache["v6"].astype(float)
    cached_pair = cache["aux__pair_all_q52"].astype(float)
    v7 = np.clip(
        np.round(0.85 * cached_v6 + 0.15 * cached_pair, 2), 0.0, 1.0
    )

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
        base_train = base.iloc[train_index].reset_index(drop=True)
        base_valid = base.iloc[valid_index].reset_index(drop=True)
        interaction_train, interaction_valid = normalized_numeric_pair(
            base_train, base_valid
        )
        ranking = select_interactions(
            interaction_train, target.iloc[train_index]
        )
        selected_rows.extend(
            {
                "fold": fold,
                "rank": rank_index + 1,
                "interaction": name,
            }
            for rank_index, name in enumerate(ranking[:100])
        )

        for profile in PROFILES:
            profile_train, profile_valid = build_profile_features(
                base_train,
                base_valid,
                interaction_train,
                interaction_valid,
                ranking,
                profile,
            )
            preprocessor = make_preprocessor(profile_train)
            train_matrix = preprocessor.fit_transform(profile_train)
            valid_matrix = preprocessor.transform(profile_valid)
            model = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                min_samples_leaf=1,
                max_features=profile.max_features,
                random_state=MODEL_SEED,
                n_jobs=1,
            )
            model.fit(train_matrix, target.iloc[train_index])
            tree_predictions = np.column_stack(
                [tree.predict(valid_matrix) for tree in model.estimators_]
            )
            for quantile in QUANTILES:
                q_name = int(round(quantile * 100))
                name = f"{profile.name}_q{q_name}"
                if name not in candidate_oof:
                    candidate_oof[name] = np.full(len(train), np.nan)
                candidate_oof[name][valid_index] = np.clip(
                    np.round(
                        np.quantile(tree_predictions, quantile, axis=1), 2
                    ),
                    0.0,
                    1.0,
                )
        print(f"fold={fold} complete", flush=True)

    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "interaction_model": "none",
            "blend_weight": 0.0,
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]
    prediction_cache: dict[str, np.ndarray] = {
        "target": target.to_numpy(dtype=float),
        "v6": cached_v6,
        "v7": v7,
    }
    for model_name, interaction_prediction in candidate_oof.items():
        prediction_cache[f"interaction__{model_name}"] = interaction_prediction
        for weight in BLEND_WEIGHTS:
            prediction = np.clip(
                np.round(
                    (1.0 - weight) * v7 + weight * interaction_prediction,
                    2,
                ),
                0.0,
                1.0,
            )
            mae = mean_absolute_error(target, prediction)
            rows.append(
                {
                    "candidate": f"{model_name}__w{weight:.3f}",
                    "interaction_model": model_name,
                    "blend_weight": weight,
                    "mae": mae,
                    "delta_vs_v7": mae - v7_mae,
                }
            )
            prediction_cache[
                f"prediction__{model_name}__w{weight:.3f}"
            ] = prediction
    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    pd.DataFrame(selected_rows).to_csv(
        args.output.with_name(args.output.stem + "_selected.csv"), index=False
    )
    if args.cache_output is not None:
        args.cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache_output, **prediction_cache)
    print(f"V7 reference: {v7_mae:.6f}")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
