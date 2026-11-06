#!/usr/bin/env python3
"""Complete-coverage JuStRank experiment reported in the paper."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from justrank_methods import (  # noqa: E402
    bradley_terry_order,
    comparison_counts,
    determinant_matrix,
    factor_borda_order,
    logodds_from_counts,
    rank_centrality_order,
    schulze_order,
)
from rank2_kemeny import (  # noqa: E402
    angular_order,
    borda_order,
    copeland_order,
    exact_dp,
    objective,
    rank2_skew_approx,
    sign_circular_order,
)


FULL_BASELINES = (
    "bradley_terry",
    "borda",
    "copeland",
    "rank_centrality",
    "schulze",
)
FACTOR_BASELINES = ("factor_borda", "factor_copeland", "factor_sign_cut")
GAIN_PATHS = [
    *(("full_gains", name) for name in FULL_BASELINES),
    *(("factor_gains_on_empirical", name) for name in FACTOR_BASELINES),
    *(("factor_gains_on_rank2", name) for name in FACTOR_BASELINES),
]


def stable_seed(seed: int, *values: object) -> int:
    payload = ":".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def ordered_questions(questions: list[str], seed: int) -> list[str]:
    return sorted(
        questions,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )


def stable_block_id(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def load_question_ids(input_path: Path) -> list[str]:
    frame = pd.read_parquet(input_path, columns=["arena_hard_question_id"])
    questions = sorted(str(value) for value in frame["arena_hard_question_id"].unique())
    if len(questions) != 500:
        raise ValueError("expected 500 Arena-Hard questions")
    return questions


def load_tokenprob_scores(
    input_path: Path, selected_questions: list[str]
) -> tuple[list[str], list[str], dict[str, np.ndarray], list[str]]:
    columns = [
        "arena_hard_question_id",
        "system",
        "judge_model",
        "judge_realization",
        "judgment_score",
    ]
    frame = pd.read_parquet(
        input_path,
        columns=columns,
        filters=[
            ("judge_realization", "==", "TokenProbs"),
            ("arena_hard_question_id", "in", selected_questions),
        ],
    ).dropna(subset=["judgment_score"])
    systems = sorted(str(value) for value in frame["system"].unique())
    questions = sorted(str(value) for value in frame["arena_hard_question_id"].unique())
    judges = sorted(str(value) for value in frame["judge_model"].unique())
    if (len(systems), len(questions), len(judges)) != (
        63,
        len(selected_questions),
        10,
    ):
        raise ValueError("unexpected JuStRank TokenProbs dimensions")
    system_index = {name: index for index, name in enumerate(systems)}
    judge_index = {name: index for index, name in enumerate(judges)}
    matrices: dict[str, np.ndarray] = {}
    for question, group in frame.groupby("arena_hard_question_id", sort=True):
        scores = np.full((len(systems), len(judges)), np.nan, dtype=float)
        for row in group.itertuples(index=False):
            scores[system_index[str(row.system)], judge_index[str(row.judge_model)]] = float(
                row.judgment_score
            )
        matrices[str(question)] = scores
    return systems, questions, matrices, judges


def blocks(questions: list[str], block_size: int) -> list[list[str]]:
    if len(questions) % block_size:
        raise ValueError("question count must be divisible by block size")
    return [
        questions[offset : offset + block_size]
        for offset in range(0, len(questions), block_size)
    ]


def build_profiles(
    matrices: dict[str, np.ndarray],
    questions: list[str],
    judges: list[str],
    block_size: int,
) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for block_number, block in enumerate(blocks(questions, block_size)):
        for judge_number, judge in enumerate(judges):
            scores = np.column_stack(
                [matrices[question][:, judge_number] for question in block]
            )
            wins, totals = comparison_counts(scores)
            profiles.append(
                {
                    "wins": wins,
                    "totals": totals,
                    "W": logodds_from_counts(wins, totals),
                    "block_number": block_number,
                    "judge": judge,
                    "block_id": stable_block_id(block),
                }
            )
    return profiles


def cycle_set(W: np.ndarray, tolerance: float = 1e-10) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for i, j, k in itertools.combinations(range(len(W)), 3):
        if W[i, j] > tolerance and W[j, k] > tolerance and W[k, i] > tolerance:
            result.add((i, j, k))
        elif W[j, i] > tolerance and W[k, j] > tolerance and W[i, k] > tolerance:
            result.add((i, k, j))
    return result


def evidence_scale(W: np.ndarray) -> float:
    return max(float(np.abs(np.triu(W, 1)).sum()), 1e-12)


def evaluate(profile: dict[str, object], cohort: np.ndarray) -> dict[str, object]:
    W = np.asarray(profile["W"])[np.ix_(cohort, cohort)]
    wins = np.asarray(profile["wins"])[np.ix_(cohort, cohort)]
    totals = np.asarray(profile["totals"])[np.ix_(cohort, cohort)]
    projected_W, factors, _ = rank2_skew_approx(W)
    model_W = determinant_matrix(factors)
    probability = expit(W)
    np.fill_diagonal(probability, 0.5)
    bt_order, bt_ok = bradley_terry_order(wins, totals)
    if not bt_ok:
        raise RuntimeError("Bradley--Terry optimization did not converge")
    full_orders = {
        "bradley_terry": bt_order,
        "borda": borda_order(W),
        "copeland": copeland_order(W),
        "rank_centrality": rank_centrality_order(probability),
        "schulze": schulze_order(probability),
    }
    factor_orders = {
        "factor_borda": factor_borda_order(factors),
        "factor_copeland": copeland_order(model_W),
        "factor_sign_cut": sign_circular_order(model_W, factors),
    }
    close = angular_order(factors)
    close_value = objective(W, close)
    close_model_value = objective(model_W, close)
    _, exact_value, _ = exact_dp(W)
    scale = evidence_scale(W)
    model_scale = evidence_scale(model_W)
    empirical_cycles = cycle_set(W)
    model_cycles = cycle_set(model_W)
    return {
        "block_id": profile["block_id"],
        "block_number": profile["block_number"],
        "judge": profile["judge"],
        "empirical_cyclic": int(bool(empirical_cycles)),
        "model_cyclic": int(bool(model_cycles)),
        "model_cyclic_triangles": len(model_cycles),
        "rank2_energy": 1.0
        - float(np.linalg.norm(W - projected_W, "fro") ** 2)
        / max(float(np.linalg.norm(W, "fro") ** 2), 1e-12),
        "relative_regret": (exact_value - close_value) / scale,
        "full_gains": {
            name: (close_value - objective(W, order)) / scale
            for name, order in full_orders.items()
        },
        "factor_gains_on_empirical": {
            name: (close_value - objective(W, order)) / scale
            for name, order in factor_orders.items()
        },
        "factor_gains_on_rank2": {
            name: (close_model_value - objective(model_W, order)) / model_scale
            for name, order in factor_orders.items()
        },
    }


def mean(records: list[dict[str, object]], key: str, name: str | None = None) -> float:
    if name is None:
        return float(np.mean([float(record[key]) for record in records]))
    return float(np.mean([float(record[key][name]) for record in records]))


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "profiles": len(records),
        "empirical_cyclic_fraction": mean(records, "empirical_cyclic"),
        "model_cyclic_fraction": mean(records, "model_cyclic"),
        "mean_model_cyclic_triangles": mean(records, "model_cyclic_triangles"),
        "mean_rank2_energy": mean(records, "rank2_energy"),
        "mean_relative_regret": mean(records, "relative_regret"),
        "mean_full_gains": {
            name: mean(records, "full_gains", name) for name in FULL_BASELINES
        },
        "mean_factor_gains_on_empirical": {
            name: mean(records, "factor_gains_on_empirical", name)
            for name in FACTOR_BASELINES
        },
        "mean_factor_gains_on_rank2": {
            name: mean(records, "factor_gains_on_rank2", name)
            for name in FACTOR_BASELINES
        },
    }


def block_bootstrap(
    records: list[dict[str, object]],
    block_universe: list[str],
    seed: int,
    repeats: int,
) -> dict[str, object]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for record in records:
        grouped[str(record["block_id"])].append(
            [float(record[family][name]) for family, name in GAIN_PATHS]
        )
    block_sums = np.asarray(
        [
            np.asarray(grouped[key], dtype=float).sum(axis=0)
            if grouped[key]
            else np.zeros(len(GAIN_PATHS), dtype=float)
            for key in block_universe
        ]
    )
    block_counts = np.asarray(
        [len(grouped[key]) for key in block_universe], dtype=float
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(block_universe), size=(repeats, len(block_universe))
    )
    sample_counts = block_counts[indices].sum(axis=1)
    valid = sample_counts > 0
    samples = block_sums[indices].sum(axis=1)[valid] / sample_counts[valid, None]
    point = block_sums.sum(axis=0) / block_counts.sum()
    alpha = 0.05 / len(GAIN_PATHS)
    results = {}
    for column, (family, name) in enumerate(GAIN_PATHS):
        results[f"{family}.{name}"] = {
            "mean": float(point[column]),
            "ci95": np.quantile(samples[:, column], [0.025, 0.975]).tolist(),
            "familywise_one_sided_lower": float(
                np.quantile(samples[:, column], alpha)
            ),
        }
    return {
        "blocks": len(block_universe),
        "profiles": len(records),
        "repeats": repeats,
        "comparisons": len(GAIN_PATHS),
        "familywise_alpha_per_comparison": alpha,
        "familywise_positive_comparisons": int(
            sum(row["familywise_one_sided_lower"] > 0.0 for row in results.values())
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproduced" / "justrank" / "justrank_complete_coverage.json",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--construction-questions", type=int, default=100)
    parser.add_argument("--evaluation-questions", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    args = parser.parse_args()

    all_questions = load_question_ids(args.input)
    question_order = ordered_questions(all_questions, args.seed)
    construction_questions = question_order[: args.construction_questions]
    evaluation_questions = question_order[-args.evaluation_questions :]
    if set(construction_questions) & set(evaluation_questions):
        raise ValueError("construction and evaluation questions must be disjoint")

    systems, loaded_construction, construction_matrices, judges = load_tokenprob_scores(
        args.input, construction_questions
    )
    if set(loaded_construction) != set(construction_questions):
        raise ValueError("construction question load mismatch")
    construction_profiles = build_profiles(
        construction_matrices,
        construction_questions,
        judges,
        args.block_size,
    )
    aggregate_W = sum(np.asarray(profile["W"]) for profile in construction_profiles)
    ability_order = np.asarray(borda_order(aggregate_W), dtype=int)
    cohorts = [row.copy() for row in ability_order.reshape(9, 7)]
    if set(np.concatenate(cohorts).tolist()) != set(range(63)):
        raise ValueError("cohorts must cover all systems exactly once")

    evaluation_systems, loaded_evaluation, evaluation_matrices, evaluation_judges = (
        load_tokenprob_scores(args.input, evaluation_questions)
    )
    if evaluation_systems != systems or evaluation_judges != judges:
        raise ValueError("construction and evaluation dimensions differ")
    if set(loaded_evaluation) != set(evaluation_questions):
        raise ValueError("evaluation question load mismatch")
    evaluation_profiles = build_profiles(
        evaluation_matrices,
        evaluation_questions,
        judges,
        args.block_size,
    )
    if len(evaluation_profiles) != 400:
        raise ValueError("expected 400 evaluation profiles per cohort")
    for cohort in cohorts:
        upper = np.triu_indices(len(cohort), 1)
        for profile in evaluation_profiles:
            totals = np.asarray(profile["totals"])[np.ix_(cohort, cohort)]
            if np.any(totals[upper] <= 0):
                raise ValueError("an evaluation cohort contains an unobserved pair")

    evaluated: list[tuple[int, list[dict[str, object]]]] = []
    for cohort_number, cohort in enumerate(cohorts, start=1):
        rows = [evaluate(profile, cohort) for profile in evaluation_profiles]
        for row in rows:
            row["cohort"] = cohort_number
        evaluated.append((cohort_number, rows))
    records = [record for _, rows in evaluated for record in rows]
    block_universe = sorted(
        {str(profile["block_id"]) for profile in evaluation_profiles}
    )
    strata = {
        "all": records,
        "empirical_cyclic": [row for row in records if row["empirical_cyclic"]],
        "empirical_acyclic": [row for row in records if not row["empirical_cyclic"]],
        "projected_model_cyclic": [row for row in records if row["model_cyclic"]],
    }
    stratum_output = {}
    for number, (name, selected) in enumerate(strata.items()):
        stratum_output[name] = {
            "summary": summarize(selected),
            "bootstrap": block_bootstrap(
                selected,
                block_universe,
                stable_seed(args.seed, "all63-ability-cohorts", name, number),
                args.bootstrap_repeats,
            ),
        }

    cohort_output = []
    for cohort_number, rows in evaluated:
        cohort = cohorts[cohort_number - 1]
        cohort_output.append(
            {
                "cohort": cohort_number,
                "ability_ranks": list(
                    range((cohort_number - 1) * 7 + 1, cohort_number * 7 + 1)
                ),
                "systems": [systems[int(index)] for index in cohort],
                "profiles": len(rows),
                "empirical_cyclic_profiles": int(
                    sum(int(row["empirical_cyclic"]) for row in rows)
                ),
            }
        )

    output = {
        "dataset": "JuStRank Arena-Hard v0.1 TokenProbs",
        "config": {
            "seed": args.seed,
            "construction_questions": len(construction_questions),
            "evaluation_questions": len(evaluation_questions),
            "block_size": args.block_size,
            "bootstrap_repeats": args.bootstrap_repeats,
        },
        "systems": len(systems),
        "judge_models": judges,
        "profiles_per_cohort": len(evaluation_profiles),
        "evaluated_profiles": len(records),
        "cohorts": cohort_output,
        "strata": stratum_output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "evaluated_profiles": output["evaluated_profiles"],
                "empirical_cyclic_profiles": len(strata["empirical_cyclic"]),
                "mean_rank2_energy": stratum_output["all"]["summary"][
                    "mean_rank2_energy"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
