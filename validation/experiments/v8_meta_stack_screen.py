# -*- coding: utf-8 -*-
"""Fast cross-fitted screen of residual stackers over cached V8 OOF features.

This is a development-only screen. Any selected recipe must later be evaluated
with fully nested base-model fitting before it can be promoted.
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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


META_SEED = 81173
SHRINKAGES = [0.25, 0.50, 0.75, 1.00]


def build_models() -> dict[str, object]:
    return {
        "quantile_a0p0001": make_pipeline(
            RobustScaler(),
            QuantileRegressor(quantile=0.50, alpha=0.0001, solver="highs"),
        ),
        "quantile_a0p0003": make_pipeline(
            RobustScaler(),
            QuantileRegressor(quantile=0.50, alpha=0.0003, solver="highs"),
        ),
        "quantile_a0p001": make_pipeline(
            RobustScaler(),
            QuantileRegressor(quantile=0.50, alpha=0.001, solver="highs"),
        ),
        "ridge_a10": make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
        "ridge_a100": make_pipeline(RobustScaler(), Ridge(alpha=100.0)),
        "hist_leaf4_l2": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=4,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=META_SEED,
        ),
        "hist_leaf8_l5": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=8,
            min_samples_leaf=80,
            l2_regularization=5.0,
            random_state=META_SEED,
        ),
        "gbr_depth1": GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=1,
            min_samples_leaf=50,
            random_state=META_SEED,
        ),
        "gbr_depth2": GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=80,
            random_state=META_SEED,
        ),
        "rf_leaf30": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=30,
            max_features=0.7,
            random_state=META_SEED,
            n_jobs=1,
        ),
        "et_leaf30": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=30,
            max_features=0.7,
            random_state=META_SEED,
            n_jobs=1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = np.load(args.cache)
    target = cache["target"].astype(float)
    v6 = cache["v6"].astype(float)
    feature_names = [
        name for name in cache.files if name.startswith(("aux__", "meta__"))
    ]
    matrix = np.column_stack([cache[name] for name in feature_names]).astype(float)
    matrix = np.column_stack([matrix, v6])
    residual_target = target - v6
    pair = cache["aux__pair_all_q52"].astype(float)
    v7 = np.clip(np.round(0.85 * v6 + 0.15 * pair, 2), 0.0, 1.0)
    v7_mae = mean_absolute_error(target, v7)

    splitter = KFold(n_splits=5, shuffle=True, random_state=META_SEED)
    rows = []
    for model_name, model in build_models().items():
        residual_oof = np.full(len(target), np.nan)
        for train_index, valid_index in splitter.split(matrix):
            model.fit(matrix[train_index], residual_target[train_index])
            residual_oof[valid_index] = model.predict(matrix[valid_index])
        for shrinkage in SHRINKAGES:
            direct = np.clip(
                np.round(v6 + shrinkage * residual_oof, 2), 0.0, 1.0
            )
            direct_mae = mean_absolute_error(target, direct)
            rows.append(
                {
                    "candidate": f"{model_name}__direct_s{shrinkage:.2f}",
                    "model": model_name,
                    "anchor": "v6",
                    "shrinkage": shrinkage,
                    "mae": direct_mae,
                    "delta_vs_v7": direct_mae - v7_mae,
                }
            )
            stack_prediction = np.clip(
                np.round(v6 + residual_oof, 2), 0.0, 1.0
            )
            v7_anchor = np.clip(
                np.round(
                    (1.0 - shrinkage) * v7
                    + shrinkage * stack_prediction,
                    2,
                ),
                0.0,
                1.0,
            )
            anchor_mae = mean_absolute_error(target, v7_anchor)
            rows.append(
                {
                    "candidate": f"{model_name}__v7_s{shrinkage:.2f}",
                    "model": model_name,
                    "anchor": "v7",
                    "shrinkage": shrinkage,
                    "mae": anchor_mae,
                    "delta_vs_v7": anchor_mae - v7_mae,
                }
            )

    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"V7 reference: {v7_mae:.6f}")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
