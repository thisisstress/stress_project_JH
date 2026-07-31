# -*- coding: utf-8 -*-
"""Stress Score Prediction V2 — Weighted Median ExtraTrees.

개발 및 검증 환경
- OS: Windows
- Python: 3.12
- numpy: 2.5.1
- pandas: 3.0.5
- scikit-learn: 1.9.0

모든 전처리기는 Train 데이터에만 fit합니다.
Test 데이터에서는 각 행 내부 연산으로만 파생변수를 생성합니다.
"""

from __future__ import annotations

import argparse
import os
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
QUANTILE = 0.50

# 원본 피처를 포함한 상태에서 추가할 복제본 수입니다.
# V1 대비 mean_working의 선택 확률은 낮추고 bone_density는 높였습니다.
FEATURE_COPY_COUNTS = {
    "mean_working": 1,
    "bmi": 2,
    "cholesterol": 2,
    "height": 2,
    "glucose": 2,
    "weight": 2,
    "cholesterol_glucose_ratio": 2,
    "bone_density": 4,
}


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

    # max_features=1인 ExtraTrees에서 피처 선택 확률을 조절합니다.
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


def median_predict(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    matrix = pipeline.named_steps["preprocessor"].transform(frame)
    forest = pipeline.named_steps["model"]
    tree_predictions = np.column_stack(
        [tree.predict(matrix) for tree in forest.estimators_]
    )
    prediction = np.quantile(tree_predictions, QUANTILE, axis=1)
    return np.clip(np.round(prediction, 2), 0.0, 1.0)


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
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(train_features)),
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
    pipeline.fit(train_features, train[TARGET])
    submission[TARGET] = median_predict(pipeline, test_features)

    output = (
        args.output_dir / "submit_v2_weighted_median_extratrees.csv"
    )
    submission.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved: {output.resolve()}")
    print(f"Rows: {len(submission):,}")


if __name__ == "__main__":
    main()
