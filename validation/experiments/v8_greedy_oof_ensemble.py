# -*- coding: utf-8 -*-
"""Greedy development-only ensemble search over saved Train OOF candidates.

This script never reads Test data.  It is only a recipe discovery step; any
recipe found here must be reconstructed and confirmed on untouched split seeds
before it can become a submission model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


RAW_BLEND_WEIGHTS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
GREEDY_WEIGHTS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def clipped_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values, 2), 0.0, 1.0)


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe-output", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--max-candidates", type=int, default=150)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    first = np.load(args.cache[0])
    target = first["target"].astype(float)
    v6 = first["v6"].astype(float)
    if "v7" in first.files:
        v7 = first["v7"].astype(float)
    else:
        pair = first["aux__pair_all_q52"].astype(float)
        v7 = clipped_round(0.85 * v6 + 0.15 * pair)
    v7_mae = mean_absolute_error(target, v7)

    direct_candidates: dict[str, np.ndarray] = {"v7_reference": v7}
    raw_candidates: dict[str, np.ndarray] = {}
    for cache_path in args.cache:
        cache = np.load(cache_path)
        if not np.array_equal(cache["target"].astype(float), target):
            raise ValueError(f"Target mismatch: {cache_path}")
        prefix = cache_path.stem
        for key in cache.files:
            values = np.asarray(cache[key], dtype=float)
            if values.shape != target.shape or not np.isfinite(values).all():
                continue
            name = f"{prefix}::{key}"
            if key.startswith("prediction__"):
                direct_candidates[name] = clipped_round(values)
            elif key.startswith(
                ("aux__", "classifier__", "interaction__", "forest__")
            ):
                raw_candidates[name] = values
            elif key.startswith("meta__") and "pair_meta" in prefix:
                raw_candidates[name] = values

    # Recreate the strongest confidence gate from the development screen.
    confidence_paths = [path for path in args.cache if path.stem == "v8_seed42_features"]
    if confidence_paths:
        confidence = np.load(confidence_paths[0])
        signal = confidence["meta__triple_std"].astype(float)
        pair = confidence["aux__pair_all_q52"].astype(float)
        lower = 0.21804926637351923
        upper = 0.27760879822071555
        weights = np.select(
            [signal <= lower, signal <= upper],
            [0.25, 0.30],
            default=0.10,
        )
        direct_candidates["confidence::triple_std_gate"] = clipped_round(
            (1.0 - weights) * v6 + weights * pair
        )

    for name, values in raw_candidates.items():
        for weight in RAW_BLEND_WEIGHTS:
            direct_candidates[f"blend_v7::{name}::w{weight:.3f}"] = clipped_round(
                (1.0 - weight) * v7 + weight * values
            )

    # Remove byte-identical predictions before ranking.
    unique: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for name, prediction in direct_candidates.items():
        key = fingerprint(prediction)
        if key in seen:
            continue
        seen.add(key)
        unique[name] = prediction

    individual_rows = []
    for name, prediction in unique.items():
        mae = mean_absolute_error(target, prediction)
        individual_rows.append(
            {
                "stage": "individual",
                "step": 0,
                "candidate": name,
                "weight": 1.0,
                "mae": mae,
                "delta_vs_v7": mae - v7_mae,
            }
        )
    individual = pd.DataFrame(individual_rows).sort_values("mae")
    selected_names = individual.head(args.max_candidates)["candidate"].tolist()
    pool = {name: unique[name] for name in selected_names}

    current = v7.copy()
    current_mae = v7_mae
    recipe: list[dict[str, object]] = []
    greedy_rows = []
    for step in range(1, args.steps + 1):
        best: tuple[float, str, float, np.ndarray] | None = None
        for name, prediction in pool.items():
            for weight in GREEDY_WEIGHTS:
                combined = clipped_round(
                    (1.0 - weight) * current + weight * prediction
                )
                mae = mean_absolute_error(target, combined)
                if best is None or mae < best[0]:
                    best = (mae, name, weight, combined)
        if best is None or best[0] >= current_mae - 1e-5:
            break
        current_mae, name, weight, current = best
        recipe.append({"candidate": name, "weight": weight})
        greedy_rows.append(
            {
                "stage": "greedy",
                "step": step,
                "candidate": name,
                "weight": weight,
                "mae": current_mae,
                "delta_vs_v7": current_mae - v7_mae,
            }
        )

    output = pd.concat(
        [individual, pd.DataFrame(greedy_rows)], ignore_index=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.recipe_output.write_text(
        json.dumps(
            {
                "v7_mae": v7_mae,
                "ensemble_mae": current_mae,
                "delta_vs_v7": current_mae - v7_mae,
                "recipe": recipe,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.cache_output is not None:
        args.cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache_output,
            target=target,
            v7=v7,
            ensemble=current,
        )
    print(f"V7 reference: {v7_mae:.6f}")
    print(f"Greedy ensemble: {current_mae:.6f}")
    print(json.dumps(recipe, ensure_ascii=False, indent=2))
    print(individual.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
