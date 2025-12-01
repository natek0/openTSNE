"""
test_ae_tsne.py

Sweep AE-TSNE over (spectral_norm, alpha, update_freq) on two datasets (S-curve, Swiss-roll),
compute all NLDR metrics (trustworthiness, LCMC, Stress, Residual Variance, Spearman ρ),
save CSV, summary plots, and also save each run's 2D scatter as PNG.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances

# 1) Import AE-TSNE and rawdata
from adaptive_eigen_tsne import AdaptiveEigenTSNE
from rawdata import get_s_curve, get_swiss_roll

# 2) Import the correct metric functions from metrics.py
from metrics import (
    stress_metric,
    residual_variance,
    spearman_rho,
    coranking_matrix,
    lcmc
)

# --------------------------
# 3) Create output directories
# --------------------------

# Directory where CSV and summary plots will go
OUTPUT_DIR = "."
# Directory where individual run embeddings will be saved
EMB_DIR = os.path.join(OUTPUT_DIR, "embeddings_png")
os.makedirs(EMB_DIR, exist_ok=True)

# --------------------------
# 4) Load datasets
# --------------------------

X_s_curve, color_s_curve = get_s_curve(n_samples=2000, noise=0.05, random_state=42)
X_swiss,    color_swiss  = get_swiss_roll(n_samples=2000, noise=0.05, random_state=42)

datasets = [
    ("S-curve",   X_s_curve,   color_s_curve),
    ("Swiss-roll",X_swiss,     color_swiss),
]

# --------------------------
# 5) Hyperparameter grids
# --------------------------

spectral_norms = ["gap", "log", "l2"]
alpha_values   = [1e-10, 1e-8, 1e-6]
update_freqs   = [10, 20, 50]

# We'll record one row per (dataset, spectral_norm, alpha, update_freq)
records = []

# --------------------------
# 6) Main sweep
# --------------------------
for name, X, color in datasets:
    # Precompute high-dimensional pairwise distances once
    D_hi = pairwise_distances(X)

    for spec_norm in spectral_norms:
        for alpha in alpha_values:
            for uf in update_freqs:

                print(f"Running {name} | norm={spec_norm}, alpha={alpha}, update_freq={uf}")

                # 6.1) Instantiate AE-TSNE with these hyperparameters
                model = AdaptiveEigenTSNE(
                    n_components=2,
                    n_neighbors=10,    # used internally for Laplacian
                    lam_init=0.1,
                    lam_max=2.0,
                    n_eigvals=8,
                    update_freq=uf,
                    rho=0.1,
                    eta_lambda=0.02,
                    beta_m=0.9,
                    beta_anneal=0.01,
                    alpha=alpha,
                    epsilon=1e-6,
                    n_iter=500,
                    learning_rate=200.0,
                    spectral_norm=spec_norm,
                    verbose=False,
                )

                # 6.2) Fit and time it
                t0 = time.time()
                Y = model.fit(X)
                elapsed = time.time() - t0

                # 6.3) Compute all metrics

                # 6.3.1) Trustworthiness (local)
                tw = trustworthiness(X, Y, n_neighbors=10)

                # 6.3.2) Pairwise distances in 2D
                D_lo = pairwise_distances(Y)

                # 6.3.3) LCMC (local; uses co‐ranking matrix)
                Q = coranking_matrix(D_hi, D_lo)
                lcmc_val = lcmc(Q, k=10)

                # 6.3.4) Stress (global)
                stress = stress_metric(D_hi, D_lo)

                # 6.3.5) Residual Variance (global)
                rv = residual_variance(D_hi, D_lo)

                # 6.3.6) Spearman ρ (global rank correlation)
                rho_val = spearman_rho(D_hi, D_lo)

                # 6.4) Save the 2D scatter of this run
                #      Filename encodes parameters
                base_fn = f"{name.replace(' ', '_')}_norm-{spec_norm}_alpha-{alpha}_uf-{uf}"
                png_path = os.path.join(EMB_DIR, base_fn + ".png")

                plt.figure(figsize=(5, 5))
                plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap="Spectral", s=5)
                plt.title(f"{name} (norm={spec_norm}, α={alpha}, uf={uf})")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"  ↳ Saved embedding plot ➜ {png_path}")

                # 6.5) Append to our records
                records.append({
                    "dataset":           name,
                    "spectral_norm":     spec_norm,
                    "alpha":             alpha,
                    "update_freq":       uf,
                    "runtime_sec":       elapsed,
                    "trustworthiness":   tw,
                    "LCMC":              lcmc_val,
                    "stress":            stress,
                    "residual_variance": rv,
                    "spearman_rho":      rho_val,
                    "embedding_png":     png_path
                })

# --------------------------
# 7) Save results to CSV
# --------------------------
df = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_DIR, "ae_tsne_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved metrics CSV ➜ {csv_path}")

# --------------------------
# 8) Summary plots
#    (same as before, but now data includes multiple metrics)
# --------------------------

# 8.1) Trustworthiness vs. update_freq
for name in df["dataset"].unique():
    subset = df[df["dataset"] == name]
    plt.figure(figsize=(8, 5))
    for spec_norm in spectral_norms:
        sub2 = subset[subset["spectral_norm"] == spec_norm]
        mean_tw = sub2.groupby("update_freq")["trustworthiness"].mean()
        plt.plot(mean_tw.index, mean_tw.values, marker="o", label=f"{spec_norm}")
    plt.title(f"Trustworthiness vs. update_freq ({name})")
    plt.xlabel("update_freq")
    plt.ylabel("Trustworthiness")
    plt.legend(title="Spectral Norm")
    plt.grid(True)
    plt.tight_layout()

    # Save summary plot
    fname_tw = f"summary_trustworthiness_{name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname_tw), dpi=300, bbox_inches="tight")
    print(f"Saved summary Trustworthiness plot ➜ {fname_tw}")
    plt.close()

# 8.2) Runtime vs. alpha
for name in df["dataset"].unique():
    subset = df[df["dataset"] == name]
    plt.figure(figsize=(8, 5))
    for spec_norm in spectral_norms:
        sub2 = subset[subset["spectral_norm"] == spec_norm]
        mean_rt = sub2.groupby("alpha")["runtime_sec"].mean()
        plt.plot(mean_rt.index, mean_rt.values, marker="o", label=f"{spec_norm}")
    plt.xscale("log")
    plt.title(f"Runtime vs. alpha ({name})")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Runtime (seconds)")
    plt.legend(title="Spectral Norm")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    # Save summary plot
    fname_rt = f"summary_runtime_{name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname_rt), dpi=300, bbox_inches="tight")
    print(f"Saved summary Runtime plot ➜ {fname_rt}")
    plt.close()

print("All done.")
