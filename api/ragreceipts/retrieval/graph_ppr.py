"""Personalized PageRank over a scipy CSR adjacency — pure, deterministic, no RNG.

HippoRAG-2's query-seeded PPR. Column-stochastic power iteration with damping
PPR_DAMPING (0.5); dangling columns redistribute uniformly so the chain is proper
and the vector stays a probability distribution. Seeds (node_id -> mass) form the
personalization vector, L1-normalized internally (uniform if empty/all-zero).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from ragreceipts.constants import PPR_DAMPING, PPR_MAX_ITER, PPR_TOL


def personalized_pagerank(
    adjacency: csr_matrix,
    seeds: dict[int, float],
    *,
    damping: float = PPR_DAMPING,
    max_iter: int = PPR_MAX_ITER,
    tol: float = PPR_TOL,
) -> np.ndarray:
    """Power-iteration PPR. Returns a length-N probability vector over node ids."""
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Column sums -> column-stochastic transition; dangling cols (sum 0) go uniform.
    col_sums = np.asarray(adjacency.sum(axis=0)).ravel()
    a = adjacency.astype(np.float64)
    dangling = col_sums == 0.0
    safe = np.where(dangling, 1.0, col_sums)
    # Scale each column by 1/colsum: M = A @ diag(1/colsum).
    m = a @ diags(1.0 / safe)

    # Personalization vector p (L1-normalized; uniform when empty/all-zero).
    p = np.zeros(n, dtype=np.float64)
    for node_id, mass in seeds.items():
        if 0 <= node_id < n:
            p[node_id] += float(mass)
    total = p.sum()
    p = np.full(n, 1.0 / n) if total <= 0.0 else p / total

    # Dangling mass is redistributed by the personalization vector each step.
    x = p.copy()
    for _ in range(max_iter):
        dangling_mass = x[dangling].sum() if dangling.any() else 0.0
        x_next = damping * (m @ x + dangling_mass * p) + (1.0 - damping) * p
        if np.abs(x_next - x).sum() < tol:
            x = x_next
            break
        x = x_next
    return x
