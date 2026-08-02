# -*- coding: utf-8 -*-
"""Stress Score Prediction V8 — Robust Subspace-Neighbor Blend.

개발 및 검증 환경
- OS: Windows
- Python: 3.14.4 (제출 권장 환경은 requirements.txt 참조)
- numpy: 2.4.4
- pandas: 3.0.3
- scikit-learn: 1.9.0

고정 모델
- V7 Pair-Neighbor Quantile Blend: 82.5%
- 3차원 부분공간 중 거리 상위 30개 이웃 중앙값: 2.5%
- 모든 4차원 부분공간 이웃 중앙값: 5.0%
- 2~4차원 부분공간 이웃 풀 48% 분위수: 7.5%
- 2~4차원 부분공간 이웃 풀 50% 분위수: 2.5%

Test 통계, Test 행 간 연산, 외부 데이터와 ID 정보는 사용하지 않습니다.
각 Test 행은 Train에서 학습한 전처리와 Train 경험적 순위에만 독립적으로
투영됩니다.
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

SUBSPACE_COLUMNS = [
    "mean_working",
    "bmi",
    "cholesterol",
    "height",
    "glucose",
    "weight",
    "cholesterol_glucose_ratio",
    "bone_density",
]


def clipped_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values, 2), 0.0, 1.0)


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
    preprocessor = make_preprocessor(weighted_train)
    train_matrix = preprocessor.fit_transform(weighted_train)
    test_matrix = preprocessor.transform(weighted_test)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=1,
        max_features=1,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(train_matrix, target)
    tree_predictions = np.column_stack(
        [tree.predict(test_matrix) for tree in model.estimators_]
    )
    return clipped_round(
        np.quantile(tree_predictions, TREE_QUANTILE, axis=1)
    )


def empirical_rank_transform(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 분포만으로 Train과 개별 Test 값을 경험적 순위로 변환."""
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


def subspace_neighbor_predictions(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, np.ndarray]:
    """2~4차원 부분공간별 최근접 Train target을 누수 없이 집계."""
    train_rank: dict[str, np.ndarray] = {}
    test_rank: dict[str, np.ndarray] = {}
    for column in SUBSPACE_COLUMNS:
        train_rank[column], test_rank[column] = empirical_rank_transform(
            train_features[column].to_numpy(dtype=float),
            test_features[column].to_numpy(dtype=float),
        )

    target_values = target.to_numpy(dtype=float)
    matrices: dict[int, np.ndarray] = {}
    triple_distances: np.ndarray | None = None
    for dimension in [2, 3, 4]:
        subsets = list(combinations(SUBSPACE_COLUMNS, dimension))
        targets = np.empty((len(test_features), len(subsets)), dtype=float)
        distances = (
            np.empty((len(test_features), len(subsets)), dtype=np.float32)
            if dimension == 3
            else None
        )
        for subset_index, subset in enumerate(subsets):
            distance = np.zeros(
                (len(test_features), len(train_features)), dtype=np.float32
            )
            for column in subset:
                distance += np.abs(
                    test_rank[column][:, None] - train_rank[column][None, :]
                )
            nearest = np.argmin(distance, axis=1)
            targets[:, subset_index] = target_values[nearest]
            if distances is not None:
                distances[:, subset_index] = distance[
                    np.arange(len(test_features)), nearest
                ]
        matrices[dimension] = targets
        if dimension == 3:
            if distances is None:
                raise RuntimeError("3차원 이웃 거리가 생성되지 않았습니다.")
            triple_distances = distances

    if triple_distances is None:
        raise RuntimeError("3차원 이웃 거리가 생성되지 않았습니다.")

    selected = np.argpartition(
        triple_distances, kth=29, axis=1
    )[:, :30]
    triple_closest30 = np.quantile(
        np.take_along_axis(matrices[3], selected, axis=1), 0.50, axis=1
    )
    pool = np.column_stack([matrices[2], matrices[3], matrices[4]])
    return {
        "pair_q52": np.quantile(matrices[2], 0.52, axis=1),
        "triple_closest30_q50": triple_closest30,
        "dim4_q50": np.quantile(matrices[4], 0.50, axis=1),
        "pool2to4_q48": np.quantile(pool, 0.48, axis=1),
        "pool2to4_q50": np.quantile(pool, 0.50, axis=1),
    }


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

    train_features = add_features(train.drop(columns=[TARGET, ID_COLUMN]))
    test_features = add_features(test.drop(columns=[ID_COLUMN]))
    v6 = tree_quantile_prediction(
        train_features, test_features, train[TARGET]
    )
    neighbor = subspace_neighbor_predictions(
        train_features, test_features, train[TARGET]
    )
    v7 = clipped_round(0.85 * v6 + 0.15 * neighbor["pair_q52"])
    submission[TARGET] = clipped_round(
        0.825 * v7
        + 0.025 * neighbor["triple_closest30_q50"]
        + 0.050 * neighbor["dim4_q50"]
        + 0.075 * neighbor["pool2to4_q48"]
        + 0.025 * neighbor["pool2to4_q50"]
    )

    output = args.output_dir / "submit_v8_robust_subspace_blend.csv"
    submission.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
