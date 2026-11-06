"""Exactness, runner-up-gap, and residual-stability experiments."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from rank2_kemeny import (  # noqa: E402
    angular_order, exact_milp, objective, rank2_margin,
)


def random_factors(rng: np.random.Generator, n: int) -> np.ndarray:
    # Continuous distribution gives general position almost surely.
    return rng.normal(size=(n, 2))


def skew_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    A = rng.normal(size=(n, n))
    E = np.triu(A, 1)
    E = E - E.T
    l1 = np.abs(np.triu(E, 1)).sum()
    return E * (scale / max(l1, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reproduced" / "synthetic"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    theorem_checks = 0
    gap_checks = 0
    rows = []
    adversarial = []

    # Exhaustive permutation check also validates the exact runner-up formula.
    for n in range(3, 10):
        reps = 80 if n <= 7 else (16 if n == 8 else 3)
        for _ in range(reps):
            h = random_factors(rng, n)
            W0 = h[:, 0, None] * h[:, 1] - h[:, 1, None] * h[:, 0]
            pi = angular_order(h)
            vals = sorted((objective(W0, p), p) for p in itertools.permutations(range(n)))
            best, second = vals[-1][0], vals[-2][0]
            delta = rank2_margin(h, pi)
            assert abs(best - objective(W0, pi)) < 1e-8
            assert abs((best - second) - 2 * delta) < 1e-7
            theorem_checks += 1
            gap_checks += 1

    # Noise sweep: measure exact recovery and the sufficient certificate.
    ratios = (0.0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.05, 1.5, 2.0, 4.0,
              8.0, 16.0, 32.0, 64.0)
    targeted_ratios = (0.5, 0.95, 1.0, 1.05, 1.5)
    for n in (8, 12, 16):
        for ratio in ratios:
            for rep in range(25):
                h = random_factors(rng, n)
                W0 = h[:, 0, None] * h[:, 1] - h[:, 1, None] * h[:, 0]
                pi0 = angular_order(h)
                delta = rank2_margin(h, pi0)
                E = skew_noise(rng, n, ratio * delta)
                W = W0 + E
                opt, optv, _ = exact_milp(W, time_limit=30)
                v0 = objective(W, pi0)
                rows.append({
                    "n": n, "ratio": ratio, "rep": rep,
                    "delta": delta,
                    "residual_l1": float(np.abs(np.triu(E, 1)).sum()),
                    "certified": int(np.abs(np.triu(E, 1)).sum() < delta),
                    "recovered": int(abs(optv - v0) < 1e-8),
                    "ll_regret": (optv - v0) / 2,
                })

        # The L1 threshold is sharp: target any pair reversed by the exact
        # minimum-loss neighbor of the angular optimum.
        for rep in range(25):
            h = random_factors(rng, n)
            W0 = h[:, 0, None] * h[:, 1] - h[:, 1, None] * h[:, 0]
            pi0 = angular_order(h)
            delta = rank2_margin(h, pi0)
            neighbors = ([pi0[1:] + pi0[:1], pi0[-1:] + pi0[:-1]] +
                         [pi0[:k] + [pi0[k + 1], pi0[k]] + pi0[k + 2:]
                          for k in range(n - 1)])
            runner = max(neighbors, key=lambda x: objective(W0, x))
            pos0 = {x: i for i, x in enumerate(pi0)}
            pos1 = {x: i for i, x in enumerate(runner)}
            a, b = next((a, b) for a in range(n) for b in range(a + 1, n)
                        if (pos0[a] < pos0[b]) != (pos1[a] < pos1[b]))
            first, second = (a, b) if pos0[a] < pos0[b] else (b, a)
            for ratio in targeted_ratios:
                E = np.zeros_like(W0)
                E[first, second] = -ratio * delta
                E[second, first] = ratio * delta
                W = W0 + E
                _, optv, _ = exact_milp(W, time_limit=30)
                v0 = objective(W, pi0)
                adversarial.append({
                    "n": n, "ratio": ratio, "rep": rep,
                    "recovered": int(abs(optv - v0) < 1e-8),
                    "strictly_flipped": int(optv - v0 > 1e-8),
                    "ll_regret_over_delta": (optv - v0) / (2 * delta),
                })

    path = args.output_dir / "synthetic_results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "config": {
            "seed": args.seed,
            "enumerated_sizes": list(range(3, 10)),
            "residual_sizes": [8, 12, 16],
            "trials_per_residual_level": 25,
            "residual_levels": list(ratios),
            "targeted_levels": list(targeted_ratios),
        },
        "theorem_checks": theorem_checks,
        "gap_checks": gap_checks,
        "sweep": {},
    }
    for n in (8, 12, 16):
        summary["sweep"][str(n)] = {}
        for ratio in sorted({x["ratio"] for x in rows}):
            xs = [x for x in rows if x["n"] == n and x["ratio"] == ratio]
            summary["sweep"][str(n)][str(ratio)] = {
                "recovery_rate": float(np.mean([x["recovered"] for x in xs])),
                "certificate_rate": float(np.mean([x["certified"] for x in xs])),
                "mean_ll_regret_over_delta": float(np.mean([
                    x["ll_regret"] / max(x["delta"], 1e-12) for x in xs])),
            }
    summary["adversarial_sharpness"] = {}
    for ratio in targeted_ratios:
        xs = [x for x in adversarial if x["ratio"] == ratio]
        summary["adversarial_sharpness"][str(ratio)] = {
            "optimal_rate": float(np.mean([x["recovered"] for x in xs])),
            "strict_flip_rate": float(np.mean([x["strictly_flipped"] for x in xs])),
            "mean_ll_regret_over_delta": float(np.mean([
                x["ll_regret_over_delta"] for x in xs])),
        }
    (args.output_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
