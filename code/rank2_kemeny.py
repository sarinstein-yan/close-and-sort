"""Exact and approximate solvers for weighted Kemeny aggregation.

Conventions: W[i,j] is the log-odds in favor of i preceding j and is skew
symmetric.  A permutation pi maximizes J(pi)=sum_{r<s} W[pi[r],pi[s]].
"""

from __future__ import annotations

import math
import time
from decimal import Decimal, localcontext
from functools import cmp_to_key
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def objective(W: np.ndarray, order: Iterable[int]) -> float:
    order = list(order)
    return float(sum(W[order[r], order[s]]
                     for r in range(len(order))
                     for s in range(r + 1, len(order))))




def rank2_skew_approx(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Best Frobenius skew-rank-two approximation and its 2D factors.

    Returns W0, h with W0[i,j]=det(h[i],h[j]), and the block singular value.
    """
    S = (np.asarray(W, dtype=float) - np.asarray(W, dtype=float).T) / 2
    vals, vecs = np.linalg.eigh(-S @ S)
    sigma = math.sqrt(max(float(vals[-1]), 0.0))
    if sigma <= 1e-14:
        return np.zeros_like(S), np.zeros((len(S), 2)), 0.0
    u = vecs[:, -1]
    v = -S @ u / sigma
    v -= u * float(u @ v)
    v /= np.linalg.norm(v)
    W0 = sigma * (np.outer(u, v) - np.outer(v, u))
    h = math.sqrt(sigma) * np.column_stack([u, v])
    return W0, h, sigma


def _robust_cross(a: np.ndarray, b: np.ndarray) -> float:
    """Cross product, with exact-on-float fallback near cancellation."""
    product_sum = abs(float(a[0] * b[1])) + abs(float(a[1] * b[0]))
    cross = float(a[0] * b[1] - a[1] * b[0])
    roundoff = 8 * np.finfo(float).eps * product_sum
    if abs(cross) > roundoff:
        return cross
    with localcontext() as context:
        context.prec = 80
        exact = (
            Decimal.from_float(float(a[0])) * Decimal.from_float(float(b[1]))
            - Decimal.from_float(float(a[1])) * Decimal.from_float(float(b[0]))
        )
    return float(exact)


def polar_order(vectors: np.ndarray) -> list[int]:
    """Counterclockwise polar order using half-planes and cross products.

    This avoids transcendental angle evaluation. Zero vectors come first;
    same-ray ties are resolved by radius and then identifier.
    """
    g = np.asarray(vectors, dtype=float)

    def compare(i: int, j: int) -> int:
        a, b = g[i], g[j]
        zero_a = bool(a[0] == 0.0 and a[1] == 0.0)
        zero_b = bool(b[0] == 0.0 and b[1] == 0.0)
        if zero_a != zero_b:
            return -1 if zero_a else 1
        if zero_a:
            return (i > j) - (i < j)

        half_a = 0 if (a[1] > 0.0 or (a[1] == 0.0 and a[0] >= 0.0)) else 1
        half_b = 0 if (b[1] > 0.0 or (b[1] == 0.0 and b[0] >= 0.0)) else 1
        if half_a != half_b:
            return -1 if half_a < half_b else 1

        cross = _robust_cross(a, b)
        if cross != 0.0:
            return -1 if cross > 0.0 else 1
        radius_a, radius_b = float(a @ a), float(b @ b)
        if radius_a != radius_b:
            return -1 if radius_a < radius_b else 1
        return (i > j) - (i < j)

    return sorted(range(len(g)), key=cmp_to_key(compare))


def angular_order(h: np.ndarray) -> list[int]:
    """Exact weighted-Kemeny order for determinant weights det(h_i,h_j)."""
    h = np.asarray(h, dtype=float)
    h0 = -h.sum(axis=0, keepdims=True)
    g = np.vstack([h0, h])
    idx = polar_order(g)
    cut = idx.index(0)
    cyc = idx[cut + 1:] + idx[:cut]
    return [i - 1 for i in cyc]


def rank2_margin(h: np.ndarray, order: Iterable[int]) -> float:
    """Exact best/runner-up log-likelihood gap delta in general position."""
    order = list(order)
    h0 = -np.asarray(h).sum(axis=0)
    g = np.vstack([h0, np.asarray(h)[order]])
    dets = [_robust_cross(g[i], g[(i + 1) % len(g)])
            for i in range(len(g))]
    return min(dets)






def exact_dp(W: np.ndarray) -> tuple[list[int], float, float]:
    """Held--Karp exact optimizer in O(n 2^n) time and O(n 2^n) memory."""
    start = time.perf_counter()
    W = np.asarray(W, dtype=float)
    n = len(W)
    N = 1 << n
    gain = np.zeros((n, N), dtype=np.float64)
    for mask in range(1, N):
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        prev = mask ^ lsb
        gain[:, mask] = gain[:, prev] + W[i, :]
    dp = np.full(N, -np.inf, dtype=np.float64)
    parent = np.full(N, -1, dtype=np.int16)
    dp[0] = 0.0
    for mask in range(1, N):
        bits = mask
        best, best_j = -np.inf, -1
        while bits:
            lsb = bits & -bits
            j = lsb.bit_length() - 1
            prev = mask ^ lsb
            val = dp[prev] + gain[j, prev]
            if val > best:
                best, best_j = val, j
            bits ^= lsb
        dp[mask], parent[mask] = best, best_j
    order, mask = [], N - 1
    while mask:
        j = int(parent[mask])
        order.append(j)
        mask ^= 1 << j
    order.reverse()
    elapsed = time.perf_counter() - start
    return order, float(dp[-1]), elapsed


def exact_milp(W: np.ndarray, time_limit: float = 120.0) -> tuple[list[int], float, float]:
    """Exact linear-ordering MILP with triangle inequalities.

    Variable x_ij for i<j is one exactly when i precedes j.  For each i<j<k,
    0 <= x_ij+x_jk-x_ik <= 1 excludes both directed 3-cycles; a tournament
    without a directed triangle is a total order.
    """
    W = np.asarray(W, dtype=float)
    n = len(W)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    col = {ij: t for t, ij in enumerate(pairs)}
    triples = [(i, j, k) for i in range(n)
               for j in range(i + 1, n) for k in range(j + 1, n)]
    A = lil_matrix((len(triples), len(pairs)), dtype=float)
    for r, (i, j, k) in enumerate(triples):
        A[r, col[i, j]] = 1
        A[r, col[j, k]] = 1
        A[r, col[i, k]] = -1
    c = np.array([-2 * W[i, j] for i, j in pairs], dtype=float)
    start = time.perf_counter()
    res = milp(c, integrality=np.ones(len(pairs)),
               bounds=Bounds(0, 1),
               constraints=LinearConstraint(A.tocsr(), 0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    elapsed = time.perf_counter() - start
    if not res.success:
        raise RuntimeError(f"MILP failed: status={res.status}, message={res.message}")
    X = np.zeros((n, n), dtype=int)
    for x, (i, j) in zip(res.x, pairs):
        if x > 0.5:
            X[i, j] = 1
        else:
            X[j, i] = 1
    order = sorted(range(n), key=lambda i: (-int(X[i].sum()), i))
    return order, objective(W, order), elapsed


def borda_order(W: np.ndarray) -> list[int]:
    scores = np.asarray(W).sum(axis=1)
    return sorted(range(len(W)), key=lambda i: (-scores[i], i))


def copeland_order(W: np.ndarray) -> list[int]:
    scores = np.sign(np.asarray(W)).sum(axis=1)
    return sorted(range(len(W)), key=lambda i: (-scores[i], i))


def sign_circular_order(W: np.ndarray, h: np.ndarray) -> list[int]:
    """Sign-only rank-two baseline in O(n^2) time.

    Rank-two MFAST is minimized by a cyclic shift of the factor circular
    order.  The first shift is scored directly.  Moving its first item i to
    the end changes the sign objective by -2 * sum_j S[i,j], so all remaining
    shifts can be scored incrementally.
    """
    h = np.asarray(h)
    cyc = polar_order(h)
    S = np.sign(np.asarray(W))
    if not cyc:
        return []

    row_sums = S.sum(axis=1)
    current_score = objective(S, cyc)
    best_order = list(cyc)
    best_key = (current_score, tuple(-i for i in best_order))

    for t in range(1, len(cyc)):
        moved = cyc[t - 1]
        current_score -= 2.0 * float(row_sums[moved])
        candidate = cyc[t:] + cyc[:t]
        candidate_key = (current_score, tuple(-i for i in candidate))
        if candidate_key > best_key:
            best_order = candidate
            best_key = candidate_key

    return best_order




