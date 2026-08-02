# -*- coding: utf-8 -*-
"""Screen ExtraTrees classification votes for the 0.01 target grid.

The target has exactly 101 legal values.  A classification forest therefore
offers a genuinely different split criterion from the regression forest while
still producing an empirical conditional target distribution.  All choices
are evaluated on Train OOF predictions only.
"""

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
from sklearn.ensemble import ExtraTreesClassifier
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


SCREEN_SEED = 42
MODEL_SEED = 42
QUANTILES = [0.48, 0.50, 0.52, 0.54, 0.56]
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
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
PROFILES = {
    "gini_mf1": {"criterion": "gini", "max_features": 1},
    "entropy_mf1": {"criterion": "entropy", "max_features": 1},
    "gini_mf2": {"criterion": "gini", "max_features": 2},
    "entropy_mf2": {"criterion": "entropy", "max_features": 2},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v7-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()

    cached = np.load(args.v7_cache)
    target = cached["target"].astype(float)
    v6 = cached["v6"].astype(float)
    pair = cached["aux__pair_all_q52"].astype(float)
    v7 = np.clip(np.round(0.85 * v6 + 0.15 * pair, 2), 0.0, 1.0)

    train = pd.read_csv(args.data_dir / "train.csv")
    if not np.array_equal(train[TARGET].to_numpy(dtype=float), target):
        raise ValueError("V7 cache target does not match train.csv")
    raw = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    weighted = apply_feature_weights(add_base_features(raw), FEATURE_COPY_COUNTS)
    groups = make_duplicate_groups(raw)
    bins = make_target_bins(pd.Series(target))
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    splits = list(splitter.split(raw, bins, groups))
    validate_splits(splits, groups, len(train))

    candidate_oof: dict[str, np.ndarray] = {}
    class_target = np.rint(target * 100).astype(int)
    for fold, (train_index, valid_index) in enumerate(splits):
        preprocessor = make_preprocessor(weighted.iloc[train_index])
        train_matrix = preprocessor.fit_transform(weighted.iloc[train_index])
        valid_matrix = preprocessor.transform(weighted.iloc[valid_index])
        for profile_name, profile in PROFILES.items():
            model = ExtraTreesClassifier(
                n_estimators=args.trees,
                min_samples_leaf=1,
                random_state=MODEL_SEED,
                n_jobs=1,
                **profile,
            )
            model.fit(train_matrix, class_target[train_index])
            votes = np.column_stack(
                [
                    model.classes_[tree.predict(valid_matrix).astype(int)] / 100.0
                    for tree in model.estimators_
                ]
            )
            for quantile in QUANTILES:
                q_name = int(round(100 * quantile))
                name = f"{profile_name}_q{q_name}"
                if name not in candidate_oof:
                    candidate_oof[name] = np.full(len(train), np.nan)
                candidate_oof[name][valid_index] = np.clip(
                    np.round(np.quantile(votes, quantile, axis=1), 2),
                    0.0,
                    1.0,
                )
        print(f"fold={fold} complete", flush=True)

    v7_mae = mean_absolute_error(target, v7)
    rows = [
        {
            "candidate": "v7_reference",
            "classifier": "none",
            "blend_weight": 0.0,
            "mae": v7_mae,
            "delta_vs_v7": 0.0,
        }
    ]
    prediction_cache: dict[str, np.ndarray] = {
        "target": target,
        "v6": v6,
        "v7": v7,
    }
    for name, classifier_prediction in candidate_oof.items():
        prediction_cache[f"classifier__{name}"] = classifier_prediction
        for anchor_name, anchor in {"v6": v6, "v7": v7}.items():
            for weight in BLEND_WEIGHTS:
                prediction = np.clip(
                    np.round(
                        (1.0 - weight) * anchor
                        + weight * classifier_prediction,
                        2,
                    ),
                    0.0,
                    1.0,
                )
                candidate = f"{anchor_name}__{name}__w{weight:.2f}"
                mae = mean_absolute_error(target, prediction)
                rows.append(
                    {
                        "candidate": candidate,
                        "classifier": name,
                        "blend_weight": weight,
                        "mae": mae,
                        "delta_vs_v7": mae - v7_mae,
                    }
                )
                prediction_cache[f"prediction__{candidate}"] = prediction

    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    if args.cache_output is not None:
        args.cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache_output, **prediction_cache)
    print(f"V7 reference: {v7_mae:.6f}")
    print(summary.head(60).to_string(index=False))


if __name__ == "__main__":
    main()
