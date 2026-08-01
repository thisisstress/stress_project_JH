# -*- coding: utf-8 -*-
"""Train-only screen of non-ExtraTrees model families.

The screen uses one fixed Development split seed. Hyperparameter selection is
allowed only here; a small fixed shortlist must later pass all Development
seeds and a fresh Audit3 set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from xgboost import XGBRegressor

from robust_validation_v5 import (
    ID_COL,
    N_SPLITS,
    TARGET,
    add_base_features,
    make_duplicate_groups,
    make_target_bins,
    validate_splits,
)


SCREEN_SEED = 42
MODEL_SEED = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameters: dict


CANDIDATES = [
    Candidate("cat_mae_d4", "catboost", {"loss_function": "MAE", "depth": 4}),
    Candidate("cat_mae_d6", "catboost", {"loss_function": "MAE", "depth": 6}),
    Candidate("cat_mae_d8", "catboost", {"loss_function": "MAE", "depth": 8}),
    Candidate(
        "cat_q52_d6",
        "catboost",
        {"loss_function": "Quantile:alpha=0.52", "depth": 6},
    ),
    Candidate("cat_rmse_d6", "catboost", {"loss_function": "RMSE", "depth": 6}),
    Candidate(
        "lgb_l1_l15",
        "lightgbm",
        {"objective": "regression_l1", "num_leaves": 15, "min_child_samples": 15},
    ),
    Candidate(
        "lgb_l1_l31",
        "lightgbm",
        {"objective": "regression_l1", "num_leaves": 31, "min_child_samples": 15},
    ),
    Candidate(
        "lgb_l1_l63",
        "lightgbm",
        {"objective": "regression_l1", "num_leaves": 63, "min_child_samples": 10},
    ),
    Candidate(
        "lgb_q52_l31",
        "lightgbm",
        {"objective": "quantile", "alpha": 0.52, "num_leaves": 31, "min_child_samples": 15},
    ),
    Candidate(
        "lgb_rmse_l31",
        "lightgbm",
        {"objective": "regression", "num_leaves": 31, "min_child_samples": 15},
    ),
    Candidate(
        "xgb_l1_d3",
        "xgboost",
        {"objective": "reg:absoluteerror", "max_depth": 3},
    ),
    Candidate(
        "xgb_l1_d5",
        "xgboost",
        {"objective": "reg:absoluteerror", "max_depth": 5},
    ),
    Candidate(
        "xgb_q50_d5",
        "xgboost",
        {"objective": "reg:quantileerror", "quantile_alpha": 0.50, "max_depth": 5},
    ),
    Candidate(
        "hist_l1_l15",
        "hist",
        {"loss": "absolute_error", "max_leaf_nodes": 15, "min_samples_leaf": 15},
    ),
    Candidate(
        "hist_l1_l31",
        "hist",
        {"loss": "absolute_error", "max_leaf_nodes": 31, "min_samples_leaf": 15},
    ),
    Candidate(
        "gbr_l1_d2",
        "gradient_boosting",
        {"loss": "absolute_error", "max_depth": 2, "min_samples_leaf": 10},
    ),
    Candidate(
        "gbr_huber_d2",
        "gradient_boosting",
        {"loss": "huber", "max_depth": 2, "min_samples_leaf": 10},
    ),
    Candidate(
        "spline_q50_k8",
        "spline_quantile",
        {"n_knots": 8, "alpha": 0.0003},
    ),
    Candidate(
        "spline_q50_k15",
        "spline_quantile",
        {"n_knots": 15, "alpha": 0.0003},
    ),
    Candidate("spline_ridge_k15", "spline_ridge", {"n_knots": 15, "alpha": 10.0}),
    Candidate("knn15_l1", "knn", {"n_neighbors": 15, "p": 1}),
    Candidate("knn30_l1", "knn", {"n_neighbors": 30, "p": 1}),
    Candidate(
        "rf_q50_mf70",
        "random_forest_quantile",
        {"quantile": 0.50, "max_features": 0.70, "min_samples_leaf": 1},
    ),
    Candidate(
        "rf_q52_mf70",
        "random_forest_quantile",
        {"quantile": 0.52, "max_features": 0.70, "min_samples_leaf": 1},
    ),
]


def make_standard_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
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
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
        ],
        sparse_threshold=0.0,
    )


def make_scaled_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
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
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", RobustScaler()),
                    ]
                ),
                numerical,
            ),
        ],
        sparse_threshold=0.0,
    )


def make_spline_preprocessor(
    frame: pd.DataFrame,
    n_knots: int,
) -> ColumnTransformer:
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
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=n_knots,
                                degree=3,
                                include_bias=False,
                                extrapolation="constant",
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numerical,
            ),
        ],
        sparse_threshold=0.0,
    )


def prepare_catboost_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    categorical = result.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    for column in categorical:
        result[column] = result[column].astype("string").fillna("missing").astype(str)
    return result, categorical


def fit_predict(
    candidate: Candidate,
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    train_target: pd.Series,
) -> np.ndarray:
    params = candidate.parameters

    if candidate.family == "catboost":
        train_frame, categorical = prepare_catboost_frame(train_features)
        valid_frame, _ = prepare_catboost_frame(valid_features)
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.035,
            l2_leaf_reg=5.0,
            random_seed=MODEL_SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            **params,
        )
        model.fit(train_frame, train_target, cat_features=categorical)
        return model.predict(valid_frame)

    if candidate.family in {
        "lightgbm",
        "xgboost",
        "hist",
        "gradient_boosting",
        "random_forest_quantile",
    }:
        preprocessor = make_standard_preprocessor(train_features)
        train_matrix = preprocessor.fit_transform(train_features)
        valid_matrix = preprocessor.transform(valid_features)

        if candidate.family == "lightgbm":
            model = LGBMRegressor(
                n_estimators=600,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                random_state=MODEL_SEED,
                n_jobs=1,
                verbosity=-1,
                **params,
            )
        elif candidate.family == "xgboost":
            model = XGBRegressor(
                n_estimators=600,
                learning_rate=0.025,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                random_state=MODEL_SEED,
                n_jobs=1,
                tree_method="hist",
                **params,
            )
        elif candidate.family == "hist":
            model = HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.05,
                l2_regularization=2.0,
                random_state=MODEL_SEED,
                **params,
            )
        elif candidate.family == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.025,
                random_state=MODEL_SEED,
                **params,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=500,
                random_state=MODEL_SEED,
                n_jobs=1,
                max_features=params["max_features"],
                min_samples_leaf=params["min_samples_leaf"],
            )

        model.fit(train_matrix, train_target)
        if candidate.family == "random_forest_quantile":
            tree_predictions = np.column_stack(
                [tree.predict(valid_matrix) for tree in model.estimators_]
            )
            return np.quantile(
                tree_predictions, params["quantile"], axis=1
            )
        return model.predict(valid_matrix)

    if candidate.family == "knn":
        pipeline = Pipeline(
            [
                ("preprocessor", make_scaled_preprocessor(train_features)),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=params["n_neighbors"],
                        weights="distance",
                        p=params["p"],
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipeline.fit(train_features, train_target)
        return pipeline.predict(valid_features)

    if candidate.family in {"spline_quantile", "spline_ridge"}:
        if candidate.family == "spline_quantile":
            model = QuantileRegressor(
                quantile=0.50,
                alpha=params["alpha"],
                solver="highs",
            )
        else:
            model = Ridge(alpha=params["alpha"])
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    make_spline_preprocessor(train_features, params["n_knots"]),
                ),
                ("model", model),
            ]
        )
        pipeline.fit(train_features, train_target)
        return pipeline.predict(valid_features)

    raise ValueError(f"알 수 없는 모델군: {candidate.family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL])
    features = add_base_features(raw_features)
    target = train[TARGET].astype(float)
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(features, target_bins, groups))
    validate_splits(splits, groups, len(train))

    selected = CANDIDATES
    if args.only:
        requested = set(args.only)
        selected = [candidate for candidate in CANDIDATES if candidate.name in requested]
        missing = requested - {candidate.name for candidate in selected}
        if missing:
            raise ValueError(f"Unknown candidates: {sorted(missing)}")

    predictions: list[dict] = []
    errors: list[dict] = []
    for candidate in selected:
        started = perf_counter()
        oof = np.full(len(train), np.nan, dtype=float)
        try:
            for fold, (train_index, valid_index) in enumerate(splits):
                fold_prediction = fit_predict(
                    candidate,
                    features.iloc[train_index],
                    features.iloc[valid_index],
                    target.iloc[train_index],
                )
                oof[valid_index] = fold_prediction
                print(
                    f"{candidate.name} fold={fold} "
                    f"mae={mean_absolute_error(target.iloc[valid_index], fold_prediction):.6f}",
                    flush=True,
                )
        except Exception as error:  # preserve failures without losing other families
            warnings.warn(f"{candidate.name} failed: {error}")
            errors.append(
                {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "error": repr(error),
                }
            )
            continue

        elapsed = perf_counter() - started
        for variant, values in {
            "raw": np.clip(oof, 0.0, 1.0),
            "round2": np.clip(np.round(oof, 2), 0.0, 1.0),
        }.items():
            name = f"{candidate.name}_{variant}"
            mae = mean_absolute_error(target, values)
            predictions.extend(
                {
                    "split_seed": SCREEN_SEED,
                    "row": int(row),
                    "candidate": name,
                    "target": float(target.iloc[row]),
                    "prediction": float(values[row]),
                }
                for row in range(len(train))
            )
            print(f"DONE {name}: {mae:.6f} ({elapsed:.1f}s)", flush=True)

    oof_frame = pd.DataFrame(predictions)
    oof_path = args.results_dir / "v7_family_screen_seed42_oof.csv"
    oof_frame.to_csv(oof_path, index=False)

    summary = (
        oof_frame.assign(
            absolute_error=lambda frame: np.abs(
                frame["target"] - frame["prediction"]
            )
        )
        .groupby("candidate", as_index=False)
        .agg(
            mean_mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            prediction_mean=("prediction", "mean"),
            prediction_std=("prediction", "std"),
        )
        .sort_values("mean_mae")
    )
    summary.to_csv(
        args.results_dir / "v7_family_screen_seed42_summary.csv",
        index=False,
    )
    (args.results_dir / "v7_family_screen_seed42_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
