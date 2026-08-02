# -*- coding: utf-8 -*-
"""Search a simple robust blend across three saved Train OOF split seeds.

Weights are restricted to non-negative 0.025 increments and a total auxiliary
weight of at most 0.50.  The search reports recipes that improve the frozen V7
on every supplied split seed; a selected recipe still requires a fresh audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


AUXILIARIES = [
    "squared_mf4_q56",
    "triple_closest30_q50",
    "triple_closest20_q50",
    "dim4_q50",
    "pool2to4_q48",
    "pool2to4_q50",
]
UNIT = 0.025
MAX_UNITS = 20


def clipped_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values, 2), 0.0, 1.0)


def load_seed42(
    base_path: Path,
    subspace_path: Path,
    criteria_path: Path,
) -> dict[str, np.ndarray]:
    base = np.load(base_path)
    subspace = np.load(subspace_path)
    criteria = np.load(criteria_path)
    target = base["target"].astype(float)
    v6 = base["v6"].astype(float)
    pair = base["aux__pair_all_q52"].astype(float)
    return {
        "target": target,
        "v7": clipped_round(0.85 * v6 + 0.15 * pair),
        "squared_mf4_q56": criteria["forest__squared_mf4_q56"].astype(float),
        "triple_closest30_q50": base["aux__triple_closest30_q50"].astype(float),
        "triple_closest20_q50": base["aux__triple_closest20_q50"].astype(float),
        "dim4_q50": subspace["aux__dim4_q50"].astype(float),
        "pool2to4_q48": subspace["aux__pool2to4_q48"].astype(float),
        "pool2to4_q50": subspace["aux__pool2to4_q50"].astype(float),
    }


def load_confirm(path: Path) -> dict[str, np.ndarray]:
    cache = np.load(path)
    target = cache["target"].astype(float)
    return {
        "target": target,
        "v7": cache["prediction__v7_reference"].astype(float),
        "squared_mf4_q56": cache["squared_mf4_q56"].astype(float),
        **{
            name: cache[f"neighbor__{name}"].astype(float)
            for name in AUXILIARIES
            if name != "squared_mf4_q56"
        },
    }


def build_weight_units(random_seed: int, random_recipes: int) -> np.ndarray:
    recipes: set[tuple[int, ...]] = {tuple([0] * len(AUXILIARIES))}
    for first in range(len(AUXILIARIES)):
        for first_units in range(1, MAX_UNITS + 1):
            values = [0] * len(AUXILIARIES)
            values[first] = first_units
            recipes.add(tuple(values))
            for second in range(first + 1, len(AUXILIARIES)):
                for second_units in range(1, MAX_UNITS - first_units + 1):
                    pair_values = values.copy()
                    pair_values[second] = second_units
                    recipes.add(tuple(pair_values))

    rng = np.random.default_rng(random_seed)
    for _ in range(random_recipes):
        total_units = int(rng.integers(1, MAX_UNITS + 1))
        concentration = float(rng.choice([0.25, 0.50, 1.0, 2.0]))
        probabilities = rng.dirichlet(
            np.full(len(AUXILIARIES), concentration)
        )
        units = rng.multinomial(total_units, probabilities)
        recipes.add(tuple(int(value) for value in units))
    return np.asarray(sorted(recipes), dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-base", type=Path, required=True)
    parser.add_argument("--seed42-subspace", type=Path, required=True)
    parser.add_argument("--seed42-criteria", type=Path, required=True)
    parser.add_argument("--confirm-cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-recipes", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=104729)
    args = parser.parse_args()

    bundles = [
        load_seed42(
            args.seed42_base,
            args.seed42_subspace,
            args.seed42_criteria,
        ),
        *[load_confirm(path) for path in args.confirm_cache],
    ]
    units = build_weight_units(args.random_seed, args.random_recipes)
    weights = units.astype(float) * UNIT
    total_weights = weights.sum(axis=1)
    seed_maes = np.empty((len(weights), len(bundles)), dtype=float)
    batch_size = 256
    for seed_index, bundle in enumerate(bundles):
        auxiliary_matrix = np.column_stack(
            [bundle[name] for name in AUXILIARIES]
        )
        for start in range(0, len(weights), batch_size):
            stop = min(start + batch_size, len(weights))
            batch_weights = weights[start:stop]
            prediction = (
                (1.0 - total_weights[start:stop])[:, None]
                * bundle["v7"][None, :]
                + batch_weights @ auxiliary_matrix.T
            )
            prediction = clipped_round(prediction)
            seed_maes[start:stop, seed_index] = np.mean(
                np.abs(prediction - bundle["target"][None, :]), axis=1
            )
    references = np.asarray(
        [
            mean_absolute_error(bundle["target"], bundle["v7"])
            for bundle in bundles
        ]
    )
    seed_deltas = seed_maes - references[None, :]
    result = pd.DataFrame(
        {
            **{
                f"weight__{name}": weights[:, index]
                for index, name in enumerate(AUXILIARIES)
            },
            "total_aux_weight": total_weights,
            "mean_mae": seed_maes.mean(axis=1),
            "mean_delta_vs_v7": seed_deltas.mean(axis=1),
            "worst_delta": seed_deltas.max(axis=1),
            "seed_win_rate": (seed_deltas < 0).mean(axis=1),
            **{
                f"seed{index}_mae": seed_maes[:, index]
                for index in range(seed_maes.shape[1])
            },
            **{
                f"seed{index}_delta": seed_deltas[:, index]
                for index in range(seed_deltas.shape[1])
            },
        }
    )
    result["all_seed_win"] = result["worst_delta"] < 0
    result = result.sort_values(
        ["all_seed_win", "mean_delta_vs_v7", "worst_delta"],
        ascending=[False, True, True],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"recipes={len(result)} all_seed_wins={int(result['all_seed_win'].sum())}")
    print(result.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
