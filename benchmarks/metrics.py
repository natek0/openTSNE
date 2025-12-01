#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metric helpers for the NLDR benchmark suite.

Provided functions
------------------
trustworthiness(X_hi, X_lo, k)
stress_metric(D_hi, D_lo)
residual_variance(D_hi, D_lo)
spearman_rho(D_hi, D_lo)
coranking_matrix(D_hi, D_lo)   -> (n×n) matrix Q
lcmc(Q, k)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness as _sk_trust
from scipy.stats import spearmanr

__all__ = [
    "trustworthiness",
    "stress_metric",
    "residual_variance",
    "spearman_rho",
    "coranking_matrix",
    "lcmc",
]

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _ensure_dist(M: ArrayLike) -> np.ndarray:
    """Return a full distance matrix (compute if M is a point cloud)."""
    A = np.asarray(M)
    if A.ndim == 2 and A.shape[0] != A.shape[1]:
        return pairwise_distances(A)
    return A


# ------------------------------------------------------------------ #
# 1. neighbourhood preservation
# ------------------------------------------------------------------ #
def trustworthiness(X_hi: ArrayLike,
                    X_lo: ArrayLike,
                    n_neighbors: int = 12) -> float:
    """Wrapper around scikit-learn’s `trustworthiness` (0–1, higher = better)."""
    return float(_sk_trust(X_hi, X_lo, n_neighbors=n_neighbors))


# ------------------------------------------------------------------ #
# 2. Kruskal stress-1
# ------------------------------------------------------------------ #
def stress_metric(D_hi: ArrayLike,
                  D_lo: ArrayLike) -> float:
    """Lower = better; 0 means perfect preservation of pairwise distances."""
    Dh = _ensure_dist(D_hi)
    Dl = _ensure_dist(D_lo)
    num = np.sum((Dh - Dl) ** 2)
    den = np.sum(Dh ** 2)
    return float(np.sqrt(num / den))


# ------------------------------------------------------------------ #
# 3. residual variance   (1 – R² of Pearson corr. on distances)
# ------------------------------------------------------------------ #
def residual_variance(D_hi: ArrayLike,
                      D_lo: ArrayLike) -> float:
    Dh = _ensure_dist(D_hi).ravel()
    Dl = _ensure_dist(D_lo).ravel()
    rho = np.corrcoef(Dh, Dl)[0, 1]
    return float(1.0 - rho ** 2)


# ------------------------------------------------------------------ #
# 4. Spearman ρ on distances
# ------------------------------------------------------------------ #
def spearman_rho(D_hi: ArrayLike,
                 D_lo: ArrayLike) -> float:
    Dh = _ensure_dist(D_hi).ravel()
    Dl = _ensure_dist(D_lo).ravel()
    rho, _ = spearmanr(Dh, Dl)
    return float(rho)


# ------------------------------------------------------------------ #
# 5. co-ranking matrix  (size n×n; row/col 0 = self)
# ------------------------------------------------------------------ #
def coranking_matrix(D_hi: ArrayLike,
                     D_lo: ArrayLike) -> np.ndarray:
    """
    Return full co-ranking matrix Q.

    Q[r, s] counts how many point-pairs have rank r in the high-dim space
    and rank s in the low-dim space.  Ranks are 0-based; row/col 0 correspond
    to the self-distance and are ignored by neighbourhood metrics such as LCMC.
    """
    Dh = _ensure_dist(D_hi)
    Dl = _ensure_dist(D_lo)
    n  = Dh.shape[0]

    # ranks: 0 for self, 1 = nearest neighbour, …
    R_hi = np.argsort(np.argsort(Dh, axis=1), axis=1)
    R_lo = np.argsort(np.argsort(Dl, axis=1), axis=1)

    Q = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            r = R_hi[i, j]
            s = R_lo[i, j]
            Q[r, s] += 1
    return Q


# ------------------------------------------------------------------ #
# 6. LCMC
# ------------------------------------------------------------------ #
def lcmc(Q: np.ndarray,
         k: int) -> float:
    """
    Local Continuity Meta-Criterion (ranks 1..k).

    LCMC = ( Σ_{r=1..k} Σ_{s=1..k} Q[r,s] ) / (k n − k²)
    """
    if k < 1:
        raise ValueError("k must be ≥ 1")
    n    = Q.shape[0]
    q_nn = Q[1 : k + 1, 1 : k + 1].sum()
    return q_nn / (k * n - k**2)
