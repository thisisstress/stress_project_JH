# -*- coding: utf-8 -*-
"""Train-only 전처리로 학습하고 submission.csv를 생성하는 최종 코드."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "stress_score"
ID_COLUMN = "ID"
MODEL_SEEDS = [42, 2026, 3407]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def add_rowwise_features(frame: pd.DataFrame) -> pd.DataFrame:
    """다른 행의 정보 없이 각 행 안의 값만 사용해 파생변수를 생성합니다."""
    result = frame.copy()
    height_m = result["height"] / 100.0
    result["bmi"] = result["weight"] / height_m.pow(2)
    result["pulse_pressure"] = (
        result["systolic_blood_pressure"] - result["diastolic_blood_pressure"]
    )
    result["mean_arterial_pressure"] = (
        result["systolic_blood_pressure"]
        + 2.0 * result["diastolic_blood_pressure"]
    ) / 3.0
    result["cholesterol_glucose_ratio"] = result["cholesterol"] / result[
        "glucose"
    ].clip(lower=1e-6)
    result["weight_height_ratio"] = result["weight"] / result["height"].clip(
        lower=1e-6
    )
    return result


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = frame.select_dtypes(exclude=np.number).columns.tolist()
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_columns,
            ),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    if TARGET not in train or ID_COLUMN not in train or ID_COLUMN not in test:
        raise ValueError("필수 컬럼이 없습니다.")
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train과 Test 입력 컬럼이 다릅니다.")
    if list(submission.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("제출 양식 컬럼이 올바르지 않습니다.")
    if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("제출 양식과 Test의 ID 순서가 다릅니다.")
    if train[ID_COLUMN].isna().any() or not train[ID_COLUMN].is_unique:
        raise ValueError("Train ID는 결측 없이 고유해야 합니다.")
    if test[ID_COLUMN].isna().any() or not test[ID_COLUMN].is_unique:
        raise ValueError("Test ID는 결측 없이 고유해야 합니다.")


def train_and_predict(
    train_features: pd.DataFrame,
    target: pd.Series,
    test_features: pd.DataFrame,
) -> np.ndarray:
    """전처리기를 Train에서만 fit하고 세 모델의 예측을 평균합니다."""
    preprocessor = make_preprocessor(train_features)
    train_matrix = preprocessor.fit_transform(train_features)
    test_matrix = preprocessor.transform(test_features)
    predictions = []
    for seed in MODEL_SEEDS:
        set_seed(seed)
        model = ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(train_matrix, target)
        predictions.append(model.predict(test_matrix))
    return np.clip(np.mean(predictions, axis=0), 0.0, 1.0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    submission = pd.read_csv(args.data_dir / "sample_submission.csv")
    validate_inputs(train, test, submission)

    train_features = add_rowwise_features(
        train.drop(columns=[ID_COLUMN, TARGET])
    )
    test_features = add_rowwise_features(test.drop(columns=[ID_COLUMN]))
    submission[TARGET] = train_and_predict(
        train_features, train[TARGET], test_features
    )
    submission.to_csv(
        args.output_dir / "submission.csv", index=False, encoding="utf-8"
    )

    environment = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "model_seeds": MODEL_SEEDS,
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
