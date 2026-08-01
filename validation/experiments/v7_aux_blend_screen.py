# -*- coding: utf-8 -*-
"""Seed-42 screen of small blends between V6 and independent auxiliary models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_V6 = "work1_bmi4_bone4_glucose3_q520_round2"
WEIGHTS = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-oof", type=Path, required=True)
    parser.add_argument("--aux-oof", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v6_long = pd.read_csv(args.v6_oof)
    v6 = (
        v6_long.loc[
            (v6_long["split_seed"] == 42)
            & (v6_long["candidate"] == FINAL_V6),
            ["row", "target", "prediction"],
        ]
        .sort_values("row")
        .rename(columns={"prediction": "v6_prediction"})
    )
    if len(v6) != 3000:
        raise ValueError("V6 Seed-42 OOF를 찾지 못했습니다.")
    reference_mae = np.abs(v6["target"] - v6["v6_prediction"]).mean()

    rows = []
    for path in args.aux_oof:
        aux_long = pd.read_csv(path)
        for name, group in aux_long.groupby("candidate"):
            aux = group[["row", "target", "prediction"]].sort_values("row")
            merged = v6.merge(aux, on=["row", "target"], how="inner")
            if len(merged) != len(v6):
                raise ValueError(f"OOF row mismatch: {path} {name}")
            for weight in WEIGHTS:
                raw = (
                    (1.0 - weight) * merged["v6_prediction"]
                    + weight * merged["prediction"]
                )
                for round2 in [False, True]:
                    prediction = np.clip(raw, 0.0, 1.0)
                    if round2:
                        prediction = np.round(prediction, 2)
                    error = np.abs(merged["target"] - prediction)
                    rows.append(
                        {
                            "auxiliary": name,
                            "weight": weight,
                            "round2": round2,
                            "mae": error.mean(),
                            "delta_vs_v6": error.mean() - reference_mae,
                            "sample_win_rate": float(
                                (
                                    error
                                    < np.abs(
                                        merged["target"]
                                        - merged["v6_prediction"]
                                    )
                                ).mean()
                            ),
                        }
                    )

    summary = pd.DataFrame(rows).sort_values("mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"V6 reference: {reference_mae:.6f}")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
