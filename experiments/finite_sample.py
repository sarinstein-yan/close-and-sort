#!/usr/bin/env python3
"""Finite-sample determinant-logit experiment reported in the paper."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rank2_kemeny import (  # noqa: E402
    angular_order,
    borda_order,
    copeland_order,
    exact_dp,
    objective,
    rank2_skew_approx,
)


def contains_majority_cycle(W: np.ndarray) -> bool:
    n = len(W)
    return any(
        W[i, j] > 0 and W[j, k] > 0 and W[k, i] > 0
        for i in range(n)
        for j in range(n)
        for k in range(n)
    )


def empirical_log_odds(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    observations: int,
) -> np.ndarray:
    n = len(probabilities)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            wins = int(rng.binomial(observations, probabilities[i, j]))
            value = math.log((wins + 0.5) / (observations - wins + 0.5))
            W[i, j], W[j, i] = value, -value
    return W


def run_seed(
    seed: int,
    *,
    instances: int,
    n: int,
    budgets: list[int],
    factor_rms: float,
    oracle_checks: int,
) -> tuple[list[dict[str, float | int | str]], int, int, int]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    cyclic_instances = 0
    checks = failures = 0

    for instance in range(instances):
        factors = rng.normal(size=(n, 2))
        factors *= factor_rms / math.sqrt(float(np.mean(factors**2)))
        W_true = (
            factors[:, 0, None] * factors[:, 1]
            - factors[:, 1, None] * factors[:, 0]
        )
        probabilities = expit(W_true)
        true_order = angular_order(factors)
        true_optimum = objective(W_true, true_order)
        is_cyclic = contains_majority_cycle(W_true)
        cyclic_instances += int(is_cyclic)

        for budget in budgets:
            W_empirical = empirical_log_odds(rng, probabilities, budget)
            W_rank2, fitted_factors, _ = rank2_skew_approx(W_empirical)
            empirical_order, _, _ = exact_dp(W_empirical)
            methods = {
                "rank2_angular": angular_order(fitted_factors),
                "empirical_exact": empirical_order,
                "borda": borda_order(W_empirical),
                "copeland": copeland_order(W_empirical),
            }
            if instance < oracle_checks:
                _, fitted_optimum, _ = exact_dp(W_rank2)
                checks += 1
                if (
                    abs(
                        objective(W_rank2, methods["rank2_angular"])
                        - fitted_optimum
                    )
                    > 1e-7
                ):
                    failures += 1

            total_energy = float(np.linalg.norm(W_empirical, "fro") ** 2)
            captured = 1.0 - float(
                np.linalg.norm(W_empirical - W_rank2, "fro") ** 2
            ) / max(total_energy, 1e-12)
            for method, order in methods.items():
                rows.append(
                    {
                        "seed": seed,
                        "instance": instance,
                        "budget": budget,
                        "method": method,
                        "true_regret": true_optimum - objective(W_true, order),
                        "rank2_energy": captured,
                    }
                )

    return rows, cyclic_instances, checks, failures


def summarize(
    rows: list[dict[str, float | int | str]],
    budgets: list[int],
) -> dict[str, object]:
    output: dict[str, object] = {}
    methods = ("rank2_angular", "empirical_exact", "borda", "copeland")
    for budget in budgets:
        selected = [row for row in rows if row["budget"] == budget]
        by_method = {
            method: [row for row in selected if row["method"] == method]
            for method in methods
        }
        condition: dict[str, object] = {}
        for method, values in by_method.items():
            condition[method] = {
                "mean_true_regret": float(
                    np.mean([float(row["true_regret"]) for row in values])
                ),
            }
        condition["rank2_energy_mean"] = float(
            np.mean(
                [float(row["rank2_energy"]) for row in by_method["rank2_angular"]]
            )
        )
        angular = np.asarray(
            [float(row["true_regret"]) for row in by_method["rank2_angular"]]
        )
        for baseline in ("empirical_exact", "borda", "copeland"):
            other = np.asarray(
                [float(row["true_regret"]) for row in by_method[baseline]]
            )
            improvement = other - angular
            condition[f"rank2_vs_{baseline}"] = {
                "mean_regret_reduction": float(improvement.mean()),
                "win_fraction": float(np.mean(improvement > 1e-10)),
            }
        output[str(budget)] = condition
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260813, 20260814, 20260815, 20260816, 20260817],
    )
    parser.add_argument("--instances-per-seed", type=int, default=1024)
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--factor-rms", type=float, default=0.65)
    parser.add_argument("--oracle-checks", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproduced" / "finite_sample_summary.json",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, float | int | str]] = []
    cyclic = checks = failures = 0
    for seed in args.seeds:
        rows, cyclic_count, check_count, failure_count = run_seed(
            seed,
            instances=args.instances_per_seed,
            n=args.n,
            budgets=args.budgets,
            factor_rms=args.factor_rms,
            oracle_checks=args.oracle_checks,
        )
        all_rows.extend(rows)
        cyclic += cyclic_count
        checks += check_count
        failures += failure_count
        print(f"seed {seed} complete", flush=True)

    if failures:
        raise AssertionError(f"{failures} rank-two oracle checks failed")
    total_profiles = len(args.seeds) * args.instances_per_seed
    result = {
        "config": {
            "seeds": args.seeds,
            "instances_per_seed": args.instances_per_seed,
            "total_latent_profiles": total_profiles,
            "n": args.n,
            "budgets": args.budgets,
            "factor_rms": args.factor_rms,
            "rank2_oracle_checks": checks,
        },
        "cyclic_fraction": cyclic / total_profiles,
        "conditions": summarize(all_rows, args.budgets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
