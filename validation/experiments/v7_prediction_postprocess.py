# -*- coding: utf-8 -*-
"""Low-dimensional post-processing audit for the independent V6 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_CANDIDATE = "work1_bmi4_bone4_glucose3_q520_round2"
SCALES = [0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20]
BIASES = [-0.010, -0.005, 0.000, 0.005, 0.010]
CENTER = 0.50


def build_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.loc[frame["candidate"] == FINAL_CANDIDATE].copy()
    rows = []
    for scale in SCALES:
        for bias in BIASES:
            adjusted = CENTER + scale * (base["prediction"] - CENTER) + bias
            for round2 in [False, True]:
                prediction = np.clip(adjusted, 0.0, 1.0)
                if round2:
                    prediction = np.round(prediction, 2)
                result = base[["split_seed", "row", "target"]].copy()
                result["candidate"] = (
                    f"scale{scale:.3f}_bias{bias:+.3f}_"
                    f"{'round2' if round2 else 'raw'}"
                )
                result["prediction"] = prediction
                rows.append(result)
    reference = base[["split_seed", "row", "target"]].copy()
    reference["candidate"] = "v6_reference"
    reference["prediction"] = base["prediction"]
    rows.append(reference)
    return pd.concat(rows, ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    detail = frame.assign(
        absolute_error=lambda data: np.abs(data["target"] - data["prediction"])
    ).groupby(["candidate", "split_seed"], as_index=False).agg(
        mae=("absolute_error", "mean")
    )
    reference = detail.loc[
        detail["candidate"] == "v6_reference",
        ["split_seed", "mae"],
    ].rename(columns={"mae": "reference_mae"})
    detail = detail.merge(reference, on="split_seed", how="left")
    detail["delta_vs_v6"] = detail["mae"] - detail["reference_mae"]
    summary = detail.groupby("candidate", as_index=False).agg(
        mean_mae=("mae", "mean"),
        seed_std=("mae", "std"),
        mean_delta_vs_v6=("delta_vs_v6", "mean"),
        seed_win_rate=("delta_vs_v6", lambda series: float((series < 0).mean())),
        worst_delta=("delta_vs_v6", "max"),
    )
    return summary.sort_values("mean_mae"), detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-oof", type=Path, required=True)
    parser.add_argument("--audit2-oof", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    development = build_candidates(pd.read_csv(args.development_oof))
    audit2 = build_candidates(pd.read_csv(args.audit2_oof))
    dev_summary, dev_detail = summarize(development)
    audit_summary, audit_detail = summarize(audit2)

    dev_summary.to_csv(
        args.results_dir / "v7_postprocess_development_summary.csv", index=False
    )
    dev_detail.to_csv(
        args.results_dir / "v7_postprocess_development_detail.csv", index=False
    )
    audit_summary.to_csv(
        args.results_dir / "v7_postprocess_audit2_summary.csv", index=False
    )
    audit_detail.to_csv(
        args.results_dir / "v7_postprocess_audit2_detail.csv", index=False
    )

    top = dev_summary.head(12)["candidate"].tolist()
    comparison = dev_summary.loc[dev_summary["candidate"].isin(top)].merge(
        audit_summary,
        on="candidate",
        suffixes=("_dev", "_audit2"),
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
