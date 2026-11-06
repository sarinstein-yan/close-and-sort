"""Frequency with which sign-only and weighted rank-two optima differ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from rank2_kemeny import angular_order, objective, sign_circular_order  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--instances", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reproduced" / "separation" / "separation_summary.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    out = {}
    for n in (4, 6, 8, 10, 16, 32):
        rows = []
        for _ in range(args.instances):
            h = rng.normal(size=(n, 2))
            W = h[:, 0, None] * h[:, 1] - h[:, 1, None] * h[:, 0]
            weighted = angular_order(h)
            sign = sign_circular_order(W, h)
            opt = objective(W, weighted)
            loss = opt - objective(W, sign)
            rows.append((loss > 1e-9, loss, loss / max(abs(opt), 1e-12)))
        out[str(n)] = {
            "instances": len(rows),
            "different_score_rate": float(np.mean([x[0] for x in rows])),
            "mean_weighted_regret": float(np.mean([x[1] for x in rows])),
            "mean_normalized_regret": float(np.mean([x[2] for x in rows])),
            "conditional_mean_normalized_regret": float(np.mean([
                x[2] for x in rows if x[0]])) if any(x[0] for x in rows) else 0.0,
        }
    result = {
        "config": {
            "seed": args.seed,
            "instances_per_size": args.instances,
            "sizes": [4, 6, 8, 10, 16, 32],
        },
        "results": out,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
