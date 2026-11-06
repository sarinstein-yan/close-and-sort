"""Utilities for the JuStRank experiment reported in the paper."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def comparison_counts(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-tie win counts and observation counts for score columns."""

    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    wins = np.zeros((n, n), dtype=float)
    totals = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            valid = np.isfinite(scores[i]) & np.isfinite(scores[j])
            total = int(valid.sum())
            if total == 0:
                continue
            left, right = scores[i, valid], scores[j, valid]
            value = float(np.sum(left > right) + 0.5 * np.sum(left == right))
            wins[i, j], wins[j, i] = value, total - value
            totals[i, j] = totals[j, i] = total
    return wins, totals


def logodds_from_counts(wins: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Construct reciprocal empirical log-odds with Jeffreys smoothing."""

    wins = np.asarray(wins, dtype=float)
    totals = np.asarray(totals, dtype=float)
    n = len(wins)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if totals[i, j] <= 0:
                continue
            value = math.log((wins[i, j] + 0.5) / (wins[j, i] + 0.5))
            W[i, j], W[j, i] = value, -value
    return W


def bradley_terry_order(
    wins: np.ndarray,
    totals: np.ndarray,
    regularization: float = 1e-4,
) -> tuple[list[int], bool]:
    """Fit Bradley--Terry scores to aggregated half-tie counts."""

    wins = np.asarray(wins, dtype=float)
    totals = np.asarray(totals, dtype=float)
    n = len(wins)
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if totals[i, j] > 0
    ]

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        scores = np.r_[x, 0.0]
        loss = 0.5 * regularization * float(x @ x)
        grad = regularization * scores
        for i, j in pairs:
            count = totals[i, j]
            target = wins[i, j]
            delta = scores[i] - scores[j]
            loss += float(count * np.logaddexp(0.0, delta) - target * delta)
            residual = count * float(expit(delta)) - target
            grad[i] += residual
            grad[j] -= residual
        return loss, grad[:-1]

    result = minimize(
        lambda x: value_grad(x),
        np.zeros(n - 1, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    scores = np.r_[result.x, 0.0]
    scores -= scores.mean()
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    converged = bool(result.success) or float(np.max(np.abs(result.jac))) < 1e-5
    return order, converged


def rank_centrality_order(probability: np.ndarray) -> list[int]:
    probability = np.asarray(probability, dtype=float)
    n = len(probability)
    transition = np.zeros((n, n), dtype=float)
    scale = max(n - 1, 1)
    for i in range(n):
        for j in range(n):
            if i != j:
                transition[i, j] = probability[j, i] / scale
        transition[i, i] = 1.0 - transition[i].sum()
    stationary = np.full(n, 1.0 / n)
    for _ in range(10_000):
        updated = stationary @ transition
        if np.max(np.abs(updated - stationary)) < 1e-13:
            stationary = updated
            break
        stationary = updated
    return sorted(range(n), key=lambda i: (-float(stationary[i]), i))


def schulze_order(probability: np.ndarray) -> list[int]:
    probability = np.asarray(probability, dtype=float)
    n = len(probability)
    strength = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and probability[i, j] > probability[j, i]:
                strength[i, j] = probability[i, j]
    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            for j in range(n):
                if j != i and j != k:
                    strength[i, j] = max(
                        strength[i, j], min(strength[i, k], strength[k, j])
                    )
    wins = np.sum(strength > strength.T, axis=1)
    margins = np.sum(strength - strength.T, axis=1)
    return sorted(
        range(n), key=lambda i: (-int(wins[i]), -float(margins[i]), i)
    )


def determinant_matrix(factors: np.ndarray) -> np.ndarray:
    factors = np.asarray(factors, dtype=float)
    return (
        factors[:, 0, None] * factors[None, :, 1]
        - factors[:, 1, None] * factors[None, :, 0]
    )


def factor_borda_order(factors: np.ndarray) -> list[int]:
    factors = np.asarray(factors, dtype=float)
    total = factors.sum(axis=0)
    scores = factors[:, 0] * total[1] - factors[:, 1] * total[0]
    return sorted(range(len(factors)), key=lambda i: (-float(scores[i]), i))
