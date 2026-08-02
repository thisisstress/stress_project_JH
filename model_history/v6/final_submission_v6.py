# -*- coding: utf-8 -*-
"""Stress Score Prediction V6 — Adaptive Feature-Probability Quantile Forest.

개발 및 검증 환경
- OS: Windows
- Python: 3.12
- numpy: 2.5.1
- pandas: 3.0.5
- scikit-learn: 1.9.0

Train-only 검증으로 고정한 설정
- ExtraTrees 1,200개, max_features=1, random_state=42
- 피처 복제 수로 무작위 피처 선택 확률 조절
- Tree 예측의 52% 분위수를 소수 둘째 자리로 반올림
- Development 평균 MAE: 0.148096
- 신규 Audit2 평균 MAE: 0.147678 (V1 대비 -0.000720, Seed 3/3 승리)

모든 전처리기는 Train 데이터에만 fit합니다.
Test에서는 각 행 내부 연산으로만 파생변수를 생성합니다.
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
QUANTILE = 0.52

# 원본 피처는 항상 포함되고 아래 수만큼 복제본을 추가합니다.
# Development에서만 구조를 선택한 뒤 완전히 새로운 Audit2 Seed에서
# 설정을 바꾸지 않고 검증했습니다.
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


def quantile_predict(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
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
    submission[TARGET] = quantile_predict(pipeline, test_features)

    output = (
        args.output_dir
        / "submit_v6_adaptive_feature_quantile_forest.csv"
    )
    submission.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
