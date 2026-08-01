from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

from robust_validation_v5 import (
    AUDIT_SEEDS,
    DEVELOPMENT_SEEDS,
    ID_COL,
    N_SPLITS,
    TARGET,
    V6_FEATURES,
    add_base_features,
    apply_feature_weights,
    make_duplicate_groups,
    make_preprocessor,
    make_target_bins,
    paired_bootstrap_interval,
    validate_splits,
)


MODEL_SEED = 42
REFERENCE_QUANTILE = 0.51
QUANTILES = [0.47, 0.49, 0.50, 0.51, 0.53]


@dataclass(frozen=True)
class ForestSpecification:
    name: str
    family: str
    min_samples_leaf: int
    bootstrap: bool
    max_samples: float | None = None


ALL_SPECIFICATIONS = [
    ForestSpecification("et_leaf1", "extra", 1, False),
    ForestSpecification("et_leaf2", "extra", 2, False),
    ForestSpecification("et_leaf3", "extra", 3, False),
    ForestSpecification("et_leaf5", "extra", 5, False),
    ForestSpecification("et_boot80_leaf1", "extra", 1, True, 0.80),
    ForestSpecification("et_boot80_leaf2", "extra", 2, True, 0.80),
    ForestSpecification("et_boot80_leaf3", "extra", 3, True, 0.80),
    ForestSpecification("rf_leaf1", "random", 1, True),
    ForestSpecification("rf_leaf2", "random", 2, True),
    ForestSpecification("rf_leaf3", "random", 3, True),
]


def make_forest(
    specification: ForestSpecification,
    n_estimators: int,
):
    common = {
        "n_estimators": n_estimators,
        "min_samples_leaf": specification.min_samples_leaf,
        "max_features": 1,
        "random_state": MODEL_SEED,
        "n_jobs": 1,
        "bootstrap": specification.bootstrap,
    }
    if specification.bootstrap and specification.max_samples is not None:
        common["max_samples"] = specification.max_samples
    if specification.family == "extra":
        return ExtraTreesRegressor(**common)
    if specification.family == "random":
        return RandomForestRegressor(**common)
    raise ValueError(f"지원하지 않는 forest family: {specification.family}")


def proximity_quantiles(
    forest,
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    valid_matrix: np.ndarray,
    quantiles: list[float],
) -> dict[float, np.ndarray]:
    """Tree proximity로 Train 타깃에 가중치를 부여해 분위수를 계산한다."""
    n_valid = valid_matrix.shape[0]
    n_train = train_matrix.shape[0]
    weights = np.zeros((n_valid, n_train), dtype=np.float32)
    all_train_indices = np.arange(n_train, dtype=np.int32)
    bootstrap_samples = None
    if forest.bootstrap:
        bootstrap_samples = forest.estimators_samples_

    for tree_index, tree in enumerate(forest.estimators_):
        if bootstrap_samples is None:
            sampled_indices = all_train_indices
        else:
            sampled_indices = np.asarray(
                bootstrap_samples[tree_index], dtype=np.int32
            )
        sampled_leaves = tree.apply(train_matrix[sampled_indices])
        valid_leaves = tree.apply(valid_matrix)
        order = np.argsort(sampled_leaves, kind="stable")
        sorted_leaves = sampled_leaves[order]
        sorted_indices = sampled_indices[order]
        unique_leaves, starts, counts = np.unique(
            sorted_leaves, return_index=True, return_counts=True
        )
        leaf_ranges = {
            leaf: (start, start + count)
            for leaf, start, count in zip(unique_leaves, starts, counts)
        }

        for leaf in np.unique(valid_leaves):
            valid_rows = np.flatnonzero(valid_leaves == leaf)
            start, stop = leaf_ranges[leaf]
            members = sorted_indices[start:stop]
            unique_members, member_counts = np.unique(
                members, return_counts=True
            )
            contribution = member_counts.astype(np.float32) / (
                len(forest.estimators_) * len(members)
            )
            weights[np.ix_(valid_rows, unique_members)] += contribution

    target_order = np.argsort(train_target, kind="stable")
    sorted_target = train_target[target_order]
    sorted_weights = weights[:, target_order]
    cumulative = np.cumsum(sorted_weights, axis=1)
    total = cumulative[:, -1]
    if not np.allclose(total, 1.0, atol=2e-4):
        raise RuntimeError("Proximity weight의 행 합이 1이 아닙니다.")

    predictions = {}
    for quantile in quantiles:
        threshold = total * quantile
        positions = np.argmax(cumulative >= threshold[:, None], axis=1)
        predictions[quantile] = sorted_target[positions]
    return predictions


def summarize(
    detail: pd.DataFrame,
    oof: pd.DataFrame,
    groups: pd.Series,
) -> pd.DataFrame:
    baseline = (
        oof[oof["candidate"] == "v1_reference"]
        .sort_values(["split_seed", "row"])
        .reset_index(drop=True)
    )
    baseline_loss = np.abs(
        baseline["target"].to_numpy() - baseline["prediction"].to_numpy()
    )
    rows = []
    for candidate in detail["candidate"].drop_duplicates():
        candidate_detail = detail[detail["candidate"] == candidate]
        candidate_oof = (
            oof[oof["candidate"] == candidate]
            .sort_values(["split_seed", "row"])
            .reset_index(drop=True)
        )
        candidate_loss = np.abs(
            candidate_oof["target"].to_numpy()
            - candidate_oof["prediction"].to_numpy()
        )
        seed_count = candidate_detail["split_seed"].nunique()
        row_loss_difference = (
            candidate_loss.reshape(seed_count, -1)
            - baseline_loss.reshape(seed_count, -1)
        ).mean(axis=0)
        ci_lower, ci_upper = paired_bootstrap_interval(
            row_loss_difference, groups
        )
        rows.append(
            {
                "candidate": candidate,
                "specification": candidate_detail["specification"].iloc[0],
                "quantile": candidate_detail["quantile"].iloc[0],
                "mean_mae": candidate_detail["mae"].mean(),
                "seed_std": candidate_detail["mae"].std(ddof=0),
                "worst_seed_mae": candidate_detail["mae"].max(),
                "best_seed_mae": candidate_detail["mae"].min(),
                "mean_delta_vs_v1": candidate_detail[
                    "delta_vs_v1"
                ].mean(),
                "seed_win_rate_vs_v1": (
                    candidate_detail["delta_vs_v1"] < 0
                ).mean(),
                "sample_win_rate_vs_v1": (
                    row_loss_difference < 0
                ).mean(),
                "paired_delta_ci95_lower": ci_lower,
                "paired_delta_ci95_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_mae", "worst_seed_mae", "paired_delta_ci95_upper"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--stage",
        choices=["screen", "development", "audit"],
        default="screen",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--config",
        nargs="+",
        choices=[item.name for item in ALL_SPECIFICATIONS],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    raw_features = train.drop(columns=[TARGET, ID_COL]).reset_index(drop=True)
    target = train[TARGET].reset_index(drop=True)
    target_array = target.to_numpy()
    groups = make_duplicate_groups(raw_features)
    target_bins = make_target_bins(target)
    features = apply_feature_weights(
        add_base_features(raw_features),
        {feature: 2 for feature in V6_FEATURES},
    )
    selected_names = set(args.config or [item.name for item in ALL_SPECIFICATIONS])
    specifications = [
        item for item in ALL_SPECIFICATIONS if item.name in selected_names
    ]
    if "et_leaf1" not in selected_names:
        specifications = [ALL_SPECIFICATIONS[0], *specifications]

    stage_seeds = {
        "screen": [DEVELOPMENT_SEEDS[0]],
        "development": DEVELOPMENT_SEEDS,
        "audit": AUDIT_SEEDS,
    }
    detail_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []

    for split_seed in stage_seeds[args.stage]:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=split_seed,
        )
        splits = list(splitter.split(raw_features, target_bins, groups))
        validate_splits(splits, groups, len(raw_features))
        candidate_predictions = {
            "v1_reference": np.zeros(len(features), dtype=float)
        }
        for specification in specifications:
            for quantile in QUANTILES:
                candidate_predictions[
                    f"{specification.name}_proximity_q{int(quantile * 100):02d}"
                ] = np.zeros(len(features), dtype=float)

        for fold, (train_index, valid_index) in enumerate(splits, start=1):
            preprocessor = make_preprocessor(features)
            train_matrix = preprocessor.fit_transform(
                features.iloc[train_index]
            )
            valid_matrix = preprocessor.transform(features.iloc[valid_index])

            for specification in specifications:
                forest = make_forest(specification, args.n_estimators)
                forest.fit(train_matrix, target.iloc[train_index])
                if specification.name == "et_leaf1":
                    tree_predictions = np.column_stack(
                        [
                            tree.predict(valid_matrix)
                            for tree in forest.estimators_
                        ]
                    )
                    candidate_predictions["v1_reference"][valid_index] = (
                        np.quantile(
                            tree_predictions, REFERENCE_QUANTILE, axis=1
                        )
                    )
                proximity = proximity_quantiles(
                    forest,
                    train_matrix,
                    target_array[train_index],
                    valid_matrix,
                    QUANTILES,
                )
                for quantile, prediction in proximity.items():
                    name = (
                        f"{specification.name}_proximity_"
                        f"q{int(quantile * 100):02d}"
                    )
                    candidate_predictions[name][valid_index] = prediction
                print(
                    f"stage={args.stage} seed={split_seed} fold={fold}/5 "
                    f"config={specification.name} complete",
                    flush=True,
                )

        reference_mae = mean_absolute_error(
            target, candidate_predictions["v1_reference"]
        )
        for candidate, prediction in candidate_predictions.items():
            if candidate == "v1_reference":
                specification_name = "et_leaf1_tree_prediction"
                quantile = REFERENCE_QUANTILE
            else:
                specification_name, quantile_text = candidate.rsplit(
                    "_proximity_q", maxsplit=1
                )
                quantile = int(quantile_text) / 100.0
            mae = mean_absolute_error(target, prediction)
            detail_rows.append(
                {
                    "stage": args.stage,
                    "split_seed": split_seed,
                    "candidate": candidate,
                    "specification": specification_name,
                    "quantile": quantile,
                    "n_estimators": args.n_estimators,
                    "mae": mae,
                    "delta_vs_v1": mae - reference_mae,
                }
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        "stage": args.stage,
                        "split_seed": split_seed,
                        "row": np.arange(len(target)),
                        "candidate": candidate,
                        "target": target,
                        "prediction": prediction,
                    }
                )
            )

    detail = pd.DataFrame(detail_rows)
    oof = pd.concat(oof_frames, ignore_index=True)
    summary = summarize(detail, oof, groups)
    prefix = f"v6_proximity_{args.stage}_{args.n_estimators}trees"
    detail.to_csv(args.results_dir / f"{prefix}_detail.csv", index=False)
    oof.to_csv(args.results_dir / f"{prefix}_oof.csv", index=False)
    summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
