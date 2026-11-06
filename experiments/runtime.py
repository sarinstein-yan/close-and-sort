#!/usr/bin/env python3
"""Single-thread runtime comparison from supplied rank-two factors."""

from __future__ import annotations

import os

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(THREAD_ENV)

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from rank2_kemeny import angular_order, exact_dp, objective  # noqa: E402


def timed_batch(function: Callable[[], object], inner: int) -> float:
    """Return steady-clock seconds per call."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(inner):
            function()
        elapsed = time.perf_counter_ns() - start
    finally:
        if was_enabled:
            gc.enable()
    return elapsed / 1e9 / inner


def exact_from_factors(factors: np.ndarray) -> list[int]:
    W = (
        factors[:, 0, None] * factors[:, 1]
        - factors[:, 1, None] * factors[:, 0]
    )
    order, _, _ = exact_dp(W)
    return order


def append_measurements(
    rows: list[dict[str, object]],
    *,
    instance_id: str,
    n: int,
    method: str,
    function: Callable[[], object],
    repeats: int,
    inner: int,
) -> None:
    for repeat in range(repeats):
        rows.append(
            {
                "suite": "scaling_factors",
                "instance_id": instance_id,
                "n": n,
                "cohort": "",
                "cyclic": "",
                "method": method,
                "repeat": repeat,
                "inner_calls": inner,
                "seconds_per_call": timed_batch(function, inner),
            }
        )


def per_instance_medians(
    rows: Iterable[dict[str, object]],
) -> dict[tuple[str, int, str], float]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["instance_id"]), int(row["n"]), str(row["method"]))
        grouped[key].append(float(row["seconds_per_call"]))
    return {key: statistics.median(values) for key, values in grouped.items()}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    medians = per_instance_medians(rows)
    output: dict[str, object] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        ids = sorted(
            {str(row["instance_id"]) for row in rows if int(row["n"]) == n}
        )
        exact = [
            medians[(item, n, "matrix_materialization_plus_exact_dp")]
            for item in ids
        ]
        close = [medians[(item, n, "closesort_from_factors")] for item in ids]
        paired = [a / b for a, b in zip(exact, close)]
        output[str(n)] = {
            "instances": len(ids),
            "exact_median_s": statistics.median(exact),
            "closesort_median_s": statistics.median(close),
            "speedup_median_paired": statistics.median(paired),
            "speedup_ratio_of_medians": statistics.median(exact)
            / statistics.median(close),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--instances", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--factor-inner", type=int, default=256)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reproduced" / "runtime"
    )
    args = parser.parse_args()
    if min(args.instances, args.repeats, args.factor_inner) < 1:
        raise ValueError("instance, repeat, and batch counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for n in range(6, 19, 2):
        factors_for_n = [rng.normal(size=(n, 2)) for _ in range(args.instances)]
        # Untimed warm-up at every dynamic-programming table size.
        exact_from_factors(factors_for_n[0])
        angular_order(factors_for_n[0])
        for index, factors in enumerate(factors_for_n):
            W = (
                factors[:, 0, None] * factors[:, 1]
                - factors[:, 1, None] * factors[:, 0]
            )
            expected = angular_order(factors)
            exact_order, exact_value, _ = exact_dp(W)
            if abs(objective(W, expected) - exact_value) > 1e-8:
                raise AssertionError(f"exactness check failed at n={n}, instance={index}")
            if set(exact_order) != set(expected):
                raise AssertionError("solver returned an invalid order")
            instance_id = f"n{n}:rep{index:02d}"
            append_measurements(
                rows,
                instance_id=instance_id,
                n=n,
                method="matrix_materialization_plus_exact_dp",
                function=lambda factors=factors: exact_from_factors(factors),
                repeats=args.repeats,
                inner=1,
            )
            append_measurements(
                rows,
                instance_id=instance_id,
                n=n,
                method="closesort_from_factors",
                function=lambda factors=factors: angular_order(factors),
                repeats=args.repeats,
                inner=args.factor_inner,
            )
        print(f"n={n} complete", flush=True)

    raw_path = args.output_dir / "runtime_benchmark.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "config": {
            "seed": args.seed,
            "sizes": list(range(6, 19, 2)),
            "instances_per_size": args.instances,
            "repeats": args.repeats,
            "factor_inner": args.factor_inner,
            "threads": 1,
            "factors_supplied": True,
        },
        "aggregation": (
            "median over repeats per instance; median over instances; speedup "
            "is the median of paired per-instance median exact/CloseSort ratios"
        ),
        "results": summarize(rows),
    }
    (args.output_dir / "runtime_benchmark_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
