# -*- coding: utf-8 -*-
"""Stress Score Prediction V7 — Pair-Neighbor Quantile Blend.

개발 및 검증 환경
- OS: Windows
- Python: 3.12
- numpy: 2.5.1
- pandas: 3.0.5
- scikit-learn: 1.9.0

Train-only 검증으로 고정한 설정
- 독립 V6 Adaptive Feature-Probability ExtraTrees: 85%
- 8개 핵심 피처의 모든 2차원 조합에서 찾은 Train 1-NN 분위수: 15%
- 최종 예측을 0.01 단위로 반올림
- Audit3: V6 0.147019, V7 0.146644, Seed 3/3 승리

Test 통계는 사용하지 않습니다. Test의 각 행은 Train에서 학습한 전처리와
Train 기준 경험적 순위에만 독립적으로 투영됩니다.
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "stress_score"
ID_COLUMN = "ID"
RANDOM_SEED = 42
N_ESTIMATORS = 1200
TREE_QUANTILE = 0.52
PAIR_QUANTILE = 0.52
PAIR_WEIGHT = 0.15

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


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """다른 행이나 Test 전체 통계를 사용하지 않는 행 단위 파생변수."""
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


def add_feature_copies(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for feature, additional_copies in FEATURE_COPY_COUNTS.items():
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


def tree_quantile_prediction(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    target: pd.Series,
) -> np.ndarray:
    weighted_train = add_feature_copies(train_features)
    weighted_test = add_feature_copies(test_features)
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(weighted_train)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=1,
                    max_features=1,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(weighted_train, target)
    matrix = pipeline.named_steps["preprocessor"].transform(weighted_test)
    forest = pipeline.named_steps["model"]
    tree_predictions = np.column_stack(
        [tree.predict(matrix) for tree in forest.estimators_]
    )
    return np.clip(
        np.round(np.quantile(tree_predictions, TREE_QUANTILE, axis=1), 2),
        0.0,
        1.0,
    )


def empirical_rank_transform(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 분포만으로 Train과 한 개씩 독립적인 Test 값을 변환."""
    median = np.nanmedian(train_values)
    train_values = np.nan_to_num(train_values, nan=median)
    test_values = np.nan_to_num(test_values, nan=median)
    sorted_train = np.sort(train_values)
    train_rank = np.searchsorted(
        sorted_train, train_values, side="right"
    ) / len(train_values)
    test_rank = np.searchsorted(
        sorted_train, test_values, side="right"
    ) / len(train_values)
    return train_rank.astype(np.float32), test_rank.astype(np.float32)


def pair_neighbor_prediction(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    target: pd.Series,
) -> np.ndarray:
    """각 2차원 피처쌍에서 가장 가까운 Train target을 분위수 집계."""
    train_rank = {}
    test_rank = {}
    for column in PAIR_COLUMNS:
        train_rank[column], test_rank[column] = empirical_rank_transform(
            train_features[column].to_numpy(dtype=float),
            test_features[column].to_numpy(dtype=float),
        )

    target_values = target.to_numpy(dtype=float)
    neighbor_targets = []
    for first, second in combinations(PAIR_COLUMNS, 2):
        distance = (
            np.abs(test_rank[first][:, None] - train_rank[first][None, :])
            + np.abs(test_rank[second][:, None] - train_rank[second][None, :])
        )
        nearest_index = np.argmin(distance, axis=1)
        neighbor_targets.append(target_values[nearest_index])
    return np.quantile(
        np.column_stack(neighbor_targets), PAIR_QUANTILE, axis=1
    )


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train과 Test의 입력 컬럼이 다릅니다.")
    if list(submission.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("제출 양식 컬럼이 올바르지 않습니다.")
    if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("제출 양식과 Test의 ID 순서가 다릅니다.")
    if train[TARGET].isna().any():
        raise ValueError("Train 타깃에 결측값이 있습니다.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    submission = pd.read_csv(args.data_dir / "sample_submission.csv")
    validate_inputs(train, test, submission)

    train_features = add_features(
        train.drop(columns=[TARGET, ID_COLUMN])
    )
    test_features = add_features(test.drop(columns=[ID_COLUMN]))
    tree_prediction = tree_quantile_prediction(
        train_features,
        test_features,
        train[TARGET],
    )
    neighbor_prediction = pair_neighbor_prediction(
        train_features,
        test_features,
        train[TARGET],
    )
    submission[TARGET] = np.clip(
        np.round(
            (1.0 - PAIR_WEIGHT) * tree_prediction
            + PAIR_WEIGHT * neighbor_prediction,
            2,
        ),
        0.0,
        1.0,
    )

    output = args.output_dir / "submit_v7_pair_neighbor_blend.csv"
    submission.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
