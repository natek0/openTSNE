#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark on the 3-D S-curve manifold.

Quick run (baselines only, colour + grid):
    python s_curve_bench.py --noise 0 --n_samples 2000 \
        --plot_dir figs_s --skip_csr
"""

from __future__ import annotations
import os, time, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 side-effect import

from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from openTSNE import TSNE

# local helpers
import metrics
import csr_callbacks as csr_cb

# --------------------------------------------------------------------------- #
# Utility: colour-aware scatter plotter
# --------------------------------------------------------------------------- #
def _plot_scatter(points : np.ndarray,
                  title  : str,
                  path   : str,
                  c      : np.ndarray | None = None,   # colour per point
                  elev   : float = 15,
                  azim   : float = -70,
                  show_axes : bool = True) -> None:
    """
    Save a scatter plot.

    • 3-D input → 3-D scatter with chosen camera view.
    • 2-D input → plain 2-D scatter.
    • `c` optional 1-D colour array (same length as points).
    """
    if points.shape[1] == 3:
        fig = plt.figure(figsize=(4, 4))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=c, cmap="viridis", s=6, alpha=0.85)
        ax.view_init(elev=elev, azim=azim)

        if show_axes:
            ax.grid(True)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        else:
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    else:  # 2-D
        plt.figure(figsize=(4, 4))
        plt.scatter(points[:, 0], points[:, 1],
                    c=c, cmap="viridis", s=8, alpha=0.85)
        plt.axis("equal")
        if show_axes:
            plt.grid(True)
            plt.xlabel("x"); plt.ylabel("y")
        else:
            plt.xticks([]); plt.yticks([])

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Metric wrapper
# --------------------------------------------------------------------------- #
def evaluate_embedding(X_hi: np.ndarray,
                       D_hi: np.ndarray,
                       Y    : np.ndarray,
                       k    : int) -> dict[str, float]:
    """Return scalar quality metrics for one embedding."""
    return {
        "trustworthiness"   : metrics.trustworthiness(X_hi, Y, n_neighbors=k),
        "stress"            : metrics.stress_metric(D_hi, Y),
        "residual_variance" : metrics.residual_variance(D_hi, Y),
        "spearman_rho"      : metrics.spearman_rho(D_hi, Y),
        "lcmc"              : metrics.lcmc(metrics.coranking_matrix(D_hi, Y), k),
    }


# --------------------------------------------------------------------------- #
# Main benchmark driver
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="S-curve NLDR benchmark")
    p.add_argument("--n_samples", type=int,   default=1500)
    p.add_argument("--noise",     type=float, default=0.05)
    p.add_argument("--n_neighbors", type=int, default=12)
    p.add_argument("--perplexity",  type=float, default=30.0)
    p.add_argument("--lam",   type=float, nargs="+", default=[0, 3, 5])
    p.add_argument("--gamma", type=float, nargs="+", default=[0.0, 0.1])
    p.add_argument("--outfile",  default="s_curve_results.csv")
    p.add_argument("--plot_dir", default="figs_s")
    p.add_argument("--skip_csr", action="store_true")
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Generate raw data  (returns X and latent parameter t)
    # ------------------------------------------------------------------ #
    X, t = datasets.make_s_curve(args.n_samples,
                                 noise=args.noise,
                                 random_state=rng)
    _plot_scatter(X, "Raw S-curve",
                  f"{args.plot_dir}/raw.png",
                  c=t)

    # ------------------------------------------------------------------ #
    # 2) Define algorithms
    # ------------------------------------------------------------------ #
    k = args.n_neighbors
    algos: dict[str, object] = {
        "Isomap"             : Isomap(n_neighbors=k, n_components=2),
        "Laplacian Eigenmaps": SpectralEmbedding(n_neighbors=k,
                                                 n_components=2,
                                                 random_state=rng),
        "LLE"                : LocallyLinearEmbedding(n_neighbors=k,
                                                      n_components=2,
                                                      method="standard",
                                                      random_state=rng),
        "t-SNE"              : TSNE(n_components=2,
                                     perplexity=args.perplexity,
                                     random_state=rng,
                                     initialization="pca"),
    }

    if not args.skip_csr:
        for lam in args.lam:
            for gamma in args.gamma:
                name = f"CSR-tSNE λ={lam} γ={gamma}"
                algos[name] = csr_cb.CSRTSNE(lam=lam,
                                             gamma=gamma,
                                             n_components=2,
                                             perplexity=args.perplexity,
                                             random_state=rng)

    # ------------------------------------------------------------------ #
    # 3) Run benchmarks
    # ------------------------------------------------------------------ #
    D_hi   = pairwise_distances(X)
    results: list[dict[str, float]] = []

    for name, method in algos.items():
        t0 = time.time()

        if hasattr(method, "fit_transform"):      # scikit-learn API
            Y = method.fit_transform(X)
        else:                                     # openTSNE API
            Y = np.asarray(method.fit(X))

        secs = time.time() - t0

        _plot_scatter(Y, name,
                      f"{args.plot_dir}/{name.replace(' ', '_')}.png",
                      c=t)

        m = evaluate_embedding(X, D_hi, Y, k)
        m.update({"method": name, "time_sec": secs})
        results.append(m)

        print(f"{name:25s}  {secs:4.1f}s  trust={m['trustworthiness']:.3f}")

    # ------------------------------------------------------------------ #
    # 4) Write CSV
    # ------------------------------------------------------------------ #
    keys = ["method", "time_sec", "trustworthiness", "stress",
            "residual_variance", "spearman_rho", "lcmc"]

    with open(args.outfile, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
