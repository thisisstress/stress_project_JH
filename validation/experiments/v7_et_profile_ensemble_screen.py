# -*- coding: utf-8 -*-
"""Development-only ensemble screen across independent ET feature profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "work1_bmi4_bone4_glucose3_q520_round2"
WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    pool = summary.loc[summary["candidate"] != REFERENCE].head(45)[
        "candidate"
    ].tolist()
    long = pd.read_csv(args.oof)
    long = long.loc[long["candidate"].isin([REFERENCE, *pool])]
    wide = long.pivot(
        index=["split_seed", "row", "target"],
        columns="candidate",
        values="prediction",
    ).reset_index()
    reference_error = np.abs(wide["target"] - wide[REFERENCE])
    rows = []
    for auxiliary in pool:
        for weight in WEIGHTS:
            raw = (1.0 - weight) * wide[REFERENCE] + weight * wide[auxiliary]
            for round2 in [False, True]:
                prediction = np.clip(raw, 0.0, 1.0)
                if round2:
                    prediction = np.round(prediction, 2)
                error = np.abs(wide["target"] - prediction)
                seed_mae = error.groupby(wide["split_seed"]).mean()
                seed_reference = reference_error.groupby(wide["split_seed"]).mean()
                seed_delta = seed_mae - seed_reference
                rows.append(
                    {
                        "auxiliary": auxiliary,
                        "weight": weight,
                        "round2": round2,
                        "mean_mae": seed_mae.mean(),
                        "mean_delta_vs_v6": seed_delta.mean(),
                        "seed_win_rate": float((seed_delta < 0).mean()),
                        "worst_seed_delta": seed_delta.max(),
                    }
                )
    result = pd.DataFrame(rows).sort_values("mean_mae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
