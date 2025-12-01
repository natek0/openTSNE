#!/usr/bin/env python3
"""
benchmark.py

Run NLDR benchmarks (baseline methods + CSR-tSNE grid) on S-curve and Swiss-roll.
Outputs 2D embeddings (PNGs) and CSV metrics (trustworthiness, LCMC, stress, RV, Spearman, runtime).
"""

import os
import time
import csv
import argparse

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE as SklearnTSNE
from adaptive_eigen_tsne import AdaptiveEigenTSNE
import umap

# Relative imports from the 'benchmarks' package
from .metrics import (
    trustworthiness,       # trustworthiness(D_hi, D_lo, k)
    coranking_matrix,      # coranking_matrix(D_hi, D_lo) -> Q
    lcmc,                  # lcmc(Q, k)
    stress_metric,         # stress_metric(D_hi, D_lo)
    residual_variance,     # residual_variance(D_hi, D_lo)
    spearman_rho,          # spearman_rho(D_hi, D_lo)
)
from .csr_tsne import CSR_TSNE  # our patched class with a proper callback

def load_dataset(
    name: str,
    n_samples: int = 2000,
    noise: float = 0.02,
    random_state: int = 42
):
    """
    Load a synthetic 3D manifold dataset (S-curve or Swiss-roll).
    Returns (X, color), where X.shape = (n_samples, 3), color.shape = (n_samples,).
    """
    if name == "s_curve":
        X, color = datasets.make_s_curve(
            n_samples, noise=noise, random_state=random_state
        )
        return X, color
    elif name == "swiss_roll":
        X, color = datasets.make_swiss_roll(
            n_samples, noise=noise, random_state=random_state
        )
        return X, color
    else:
        raise ValueError(f"Unknown dataset: {name}")


def compute_metrics(X_high: np.ndarray, Y_low: np.ndarray, k: int = 10):
    """
    Compute (Trustworthiness T_k, LCMC_k, Stress, Residual Variance, Spearman rho)
    between the high-dimensional X_high and low-dimensional Y_low, using exactly k neighbors.
    """
    # Pairwise distance matrices:
    D_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=2)
    D_low = np.linalg.norm(Y_low[:, None, :] - Y_low[None, :, :], axis=2)

    # Trustworthiness T_k:
    T_k = trustworthiness(X_high, Y_low, n_neighbors=k)

    # Coranking and LCMC:
    Q = coranking_matrix(D_high, D_low)
    L_k = lcmc(Q, k)

    # Stress (Kruskal):
    stress_val = stress_metric(D_high, D_low)

    # Residual Variance:
    rv = residual_variance(D_high, D_low)

    # Spearman correlation:
    rho = spearman_rho(D_high, D_low)

    return T_k, L_k, stress_val, rv, rho


def save_embedding_plot(Y: np.ndarray, color: np.ndarray, out_path: str):
    """
    Save a 2D scatterplot of Y (colored by 'color') to out_path (PNG).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap="Spectral", s=5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_single_method(
    name: str,
    model,
    X: np.ndarray,
    color: np.ndarray,
    plot_dir: str,
    k: int = 10
):
    """
    Fit 'model' to X, get 2D embedding Y, compute metrics, save plot.
    Returns: (name, T_k, LCMC_k, stress, rv, rho, runtime).
    """
    start_time = time.time()
    if hasattr(model, "fit_transform"):
        Y = model.fit_transform(X)
    else:
        model.fit(X)
        Y = model.transform(X)
    runtime = time.time() - start_time

    # Compute metrics exactly with k neighbors
    T_k, L_k, stress_val, rv, rho = compute_metrics(X, Y, k=k)

    # Save PNG
    os.makedirs(plot_dir, exist_ok=True)
    safe_name = name.replace("=", "").replace(",", "_").replace(" ", "_")
    png_path = os.path.join(plot_dir, f"{safe_name}.png")
    save_embedding_plot(Y, color, png_path)

    return (name, T_k, L_k, stress_val, rv, rho, runtime)


def main():
    parser = argparse.ArgumentParser(
        description="Run NLDR benchmarks: baseline + CSR-tSNE (normalized Laplacian)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["s_curve", "swiss_roll"],
        help="Choose dataset: 's_curve' or 'swiss_roll'."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        required=True,
        help="Directory to save embedding PNGs (will be created)."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="CSV file to write metrics to (will be created/truncated)."
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
        help="Fixed k for k-NN graphs (both baseline methods and CSR)."
    )

    # Baseline + optionally one CSR
    parser.add_argument(
        "--with_csr",
        action="store_true",
        help="Append one CSR-tSNE variant (using --lam, --gamma)."
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.0,
        help="Spectral penalty λ for CSR-tSNE (only if --with_csr)."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Curvature penalty γ for CSR-tSNE (only if --with_csr)."
    )

    parser.add_argument(
        "--with_ae",
        action="store_true",
        help="Append one AdaptiveEigenTSNE variant (using its own λ‐schedule)."
    )

    parser.add_argument(
        "--lam_init",
        type=float,
        default=0.1,
        help="Initial λ for AE-TSNE (only if --with_ae)."
    )
    parser.add_argument(
        "--lam_max",
        type=float,
        default=2.0,
        help="Maximum λ for AE-TSNE annealing (only if --with_ae)."
    )
    parser.add_argument(
        "--n_eigvals",
        type=int,
        default=8,
        help="Number of eigenvalues m for AE-TSNE. (only if --with_ae)."
    )

    parser.add_argument(
        "--update_freq",
        type=int,
        default=20,
        help="How many t-SNE iterations between spectrum updates (only if --with_ae)."
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=0.1,
        help="Target gradient ratio for AE-TSNE’s λ adaptation (only if --with_ae)."
    )

    parser.add_argument(
        "--eta_lambda",
        type=float,
        default=0.02,
        help="Learning rate for λ adaptation (only if --with_ae)."
    )

    parser.add_argument(
        "--beta_m",
        type=float,
        default=0.9,
        help="Momentum coefficient (β_m) for λ updates (only if --with_ae)."
    )

    parser.add_argument(
        "--beta_anneal",
        type=float,
        default=0.01,
        help="Annealing rate (β_anneal) for λ_base(t) (only if --with_ae)."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-8,
        help="Regularization weight α on ||λ̂^Y||² (only if --with_ae)."
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Numerical regularizer ε for Laplacian (only if --with_ae)."
    )


    # CSR-only mode (grid of λ,γ)
    parser.add_argument(
        "--csr_only",
        action="store_true",
        help="Skip baselines and run only CSR-tSNE variants (--csr_params)."
    )
    parser.add_argument(
        "--csr_params",
        nargs=2,
        metavar=("LAM", "GAMMA"),
        action="append",
        help="Specify one CSR-tSNE variant as '<lam> <gamma>'. Repeat to get multiple."
    )

    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(args.plot_dir, exist_ok=True)
    out_parent = os.path.dirname(args.outfile)
    if out_parent and not os.path.exists(out_parent):
        os.makedirs(out_parent, exist_ok=True)

    # Build list of (name, model) to run
    algos = []

    if args.csr_only:
        if not args.csr_params:
            parser.error("--csr_only requires at least one --csr_params <lam> <gamma>.")
        for lam_str, gamma_str in args.csr_params:
            lam_val = float(lam_str)
            gamma_val = float(gamma_str)
            model = CSR_TSNE(
                n_components=2,
                n_neighbors=args.n_neighbors,   # <— guarantee k=10 here
                lam=lam_val,
                gamma=gamma_val,
                random_state=42,
                # e.g. perplexity=30.0, learning_rate=200.0 if desired
            )
            name = f"CSR-λ{lam_val}-γ{gamma_val}"
            algos.append((name, model))
    else:
        # Baseline methods (no CSR)
        algos = [
            ("PCA", PCA(n_components=2, random_state=42)),

            (
                "Isomap",
                Isomap(
                    n_neighbors=args.n_neighbors,
                    n_components=2,
                    n_jobs=-1,
                )
            ),

            (
                "Laplacian_Eigenmaps",
                SpectralEmbedding(
                    n_neighbors=args.n_neighbors,
                    n_components=2,
                    eigen_solver="arpack",
                    affinity="nearest_neighbors",  # Explicitly set affinity
                    random_state=42,
                )
            ),

            (
                "LLE",
                LocallyLinearEmbedding(
                    n_neighbors=args.n_neighbors,
                    n_components=2,
                    method="standard",
                    eigen_solver="auto",  # Explicitly set eigen solver
                    n_jobs=-1,
                    random_state=43,  # Different random state from Laplacian
                )
            ),

            (
                "t-SNE",
                SklearnTSNE(
                    n_components=2,
                    perplexity=30.0,
                    random_state=42,
                    n_jobs=-1,
                )
            ),

            ("UMAP", umap.UMAP(
                n_neighbors=args.n_neighbors, 
                min_dist=0.1,
                random_state=42
            )),
        ]

        if args.with_csr:
            model = CSR_TSNE(
                n_components=2,
                n_neighbors=args.n_neighbors,   # <— guarantee k=10 here too
                lam=args.lam,
                gamma=args.gamma,
                random_state=42,
            )
            name = f"CSR-λ{args.lam}-γ{args.gamma}"
            algos.append((name, model))

        if args.with_ae:
            model = AdaptiveEigenTSNE(
                n_components=2,
                n_neighbors=args.n_neighbors,
                lam_init=args.lam_init,
                lam_max=args.lam_max,
                n_eigvals=args.n_eigvals,
                update_freq=args.update_freq,
                rho=args.rho,
                eta_lambda=args.eta_lambda,
                beta_m=args.beta_m,
                beta_anneal=args.beta_anneal,
                alpha=args.alpha,
                epsilon=args.epsilon,
                random_state=42,
        )
        algos.append(
            (
                f"AE-TSNE(λ₀={args.lam_init:.2f},λ_max={args.lam_max:.2f})",
                model,
            )
        )

    # Load data once
    X, color = load_dataset(
        args.dataset, n_samples=2000, noise=0.02, random_state=42
    )

    # Write CSV header
    with open(args.outfile, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "Method",
                "Trustworthiness",
                "LCMC10",
                "Stress",
                "RV",
                "Spearman",
                "Runtime",
            ]
        )

        # Run each algorithm
        for name, model in algos:
            print(f"Running {name:30}", end="", flush=True)
            (m_name, T_k, L_k, stress_val, rv, rho, runtime) = run_single_method(
                name=name,
                model=model,
                X=X,
                color=color,
                plot_dir=args.plot_dir,
                k=args.n_neighbors,
            )

            # Format floats to 5 decimal places
            writer.writerow(
                [
                    m_name,
                    f"{T_k:.5f}",
                    f"{L_k:.5f}",
                    f"{stress_val:.5f}",
                    f"{rv:.5f}",
                    f"{rho:.5f}",
                    f"{runtime:.5f}",
                ]
            )

            print(f"  done (T={T_k:.3f}, LCMC={L_k:.3f}, stress={stress_val:.2f})")


if __name__ == "__main__":
    main()