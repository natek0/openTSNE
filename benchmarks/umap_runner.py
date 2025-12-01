# benchmarks/umap_runner.py
"""
Light-weight wrapper that runs UMAP on a data matrix *X* and computes
the same metrics your project already logs (trustworthiness, LCMC, stress,
residual variance, Spearman ρ).

Usage inside benchmark driver:
    from umap_runner import run_umap
    Y, metric_dict = run_umap(X, metric_funcs, n_neighbors=15, min_dist=0.1)
"""

from __future__ import annotations
import time
from typing import Dict, Callable, Sequence

import numpy as np
import umap  # pip install umap-learn


def run_umap(
    X: np.ndarray,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Run UMAP and return the 2-D embedding plus a dict of evaluation metrics."""
    t0 = time.time()
    model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )
    Y = model.fit_transform(X)
    runtime = time.time() - t0

    # ----- compute metrics in the exact same way your tsne scripts do -----
    scores = {name: fn(X, Y) for name, fn in metric_funcs.items()}
    scores["runtime_sec"] = runtime
    scores["method"] = "umap"
    scores["n_neighbors"] = n_neighbors
    scores["min_dist"] = min_dist
    return Y, scores
