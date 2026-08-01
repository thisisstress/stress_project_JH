# -*- coding: utf-8 -*-
"""Audit a fixed 80:20 blend of V6 q52 and same-forest q49.5 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import pandas as pd

from robust_validation_v5 import ID_COL, TARGET, make_duplicate_groups


Q52 = "work1_bmi4_bone4_glucose3_q520_round2"
Q495 = "work1_bmi4_bone4_glucose3_q495_round2"
Q495_WEIGHT = 0.20
BOOTSTRAP_SEED = 20260802


def cluster_interval(
    row_delta: np.ndarray,
    groups: pd.Series,
    draws: int = 10_000,
) -> tuple[float, float]:
    table = pd.DataFrame(
        {"group": groups.to_numpy(), "delta": row_delta}
    ).groupby("group", as_index=False).agg(
        delta_sum=("delta", "sum"),
        row_count=("delta", "size"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(draws)
    group_count = len(table)
    sums = table["delta_sum"].to_numpy()
    counts = table["row_count"].to_numpy()
    for draw in range(draws):
        sampled = rng.integers(0, group_count, size=group_count)
        estimates[draw] = sums[sampled].sum() / counts[sampled].sum()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def evaluate(
    q52_path: Path,
    q495_path: Path,
    groups: pd.Series,
    stage: str,
) -> tuple[pd.DataFrame, dict]:
    q52 = pd.read_csv(q52_path)
    q495 = pd.read_csv(q495_path)
    q52 = q52.loc[q52["candidate"] == Q52]
    q495 = q495.loc[q495["candidate"] == Q495]
    merged = q52.merge(
        q495,
        on=["split_seed", "row", "target"],
        suffixes=("_q52", "_q495"),
        validate="one_to_one",
    )
    merged["blend_prediction"] = np.clip(
        np.round(
            (1.0 - Q495_WEIGHT) * merged["prediction_q52"]
            + Q495_WEIGHT * merged["prediction_q495"],
            2,
        ),
        0.0,
        1.0,
    )
    merged["q52_error"] = np.abs(
        merged["target"] - merged["prediction_q52"]
    )
    merged["blend_error"] = np.abs(
        merged["target"] - merged["blend_prediction"]
    )
    detail = merged.groupby("split_seed", as_index=False).agg(
        q52_mae=("q52_error", "mean"),
        blend_mae=("blend_error", "mean"),
    )
    detail["delta_vs_q52"] = detail["blend_mae"] - detail["q52_mae"]
    row_delta = (
        merged.assign(delta=merged["blend_error"] - merged["q52_error"])
        .groupby("row")["delta"]
        .mean()
        .reindex(range(len(groups)))
        .to_numpy()
    )
    ci_lower, ci_upper = cluster_interval(row_delta, groups)
    summary = {
        "stage": stage,
        "q52_mean_mae": detail["q52_mae"].mean(),
        "blend_mean_mae": detail["blend_mae"].mean(),
        "mean_delta_vs_q52": detail["delta_vs_q52"].mean(),
        "seed_win_rate": float((detail["delta_vs_q52"] < 0).mean()),
        "worst_seed_delta": detail["delta_vs_q52"].max(),
        "paired_ci95_lower": ci_lower,
        "paired_ci95_upper": ci_upper,
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--development-q52", type=Path, required=True)
    parser.add_argument("--development-q495", type=Path, required=True)
    parser.add_argument("--audit2-q52", type=Path, required=True)
    parser.add_argument("--audit2-q495", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    groups = make_duplicate_groups(train.drop(columns=[TARGET, ID_COL]))
    dev_detail, dev_summary = evaluate(
        args.development_q52,
        args.development_q495,
        groups,
        "development",
    )
    audit_detail, audit_summary = evaluate(
        args.audit2_q52,
        args.audit2_q495,
        groups,
        "audit2",
    )
    pd.concat(
        [
            dev_detail.assign(stage="development"),
            audit_detail.assign(stage="audit2"),
        ],
        ignore_index=True,
    ).to_csv(args.results_dir / "v7_q52_q495_blend_detail.csv", index=False)
    summary = pd.DataFrame([dev_summary, audit_summary])
    summary.to_csv(
        args.results_dir / "v7_q52_q495_blend_summary.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
