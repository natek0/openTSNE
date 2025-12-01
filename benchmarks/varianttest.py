# benchmarks/varianttest.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import importlib.util

# ─── 1) Minimal dataset loader (smaller sample) ───────────────────────
def load_small_dataset(name, n_samples=500, noise=0.02, random_state=42):
    if name == "s_curve":
        X, labels = datasets.make_s_curve(n_samples, noise=noise, random_state=random_state)
        return X, labels
    elif name == "swiss_roll":
        X, labels = datasets.make_swiss_roll(n_samples, noise=noise, random_state=random_state)
        return X, labels
    else:
        raise ValueError(f"Unknown dataset: {name}")

# ─── 2) Dynamically import CSR_TSNE from the correct path ──────────────
spec = importlib.util.spec_from_file_location("csr_tsne", "benchmarks/csr_tsne.py")
csr_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csr_mod)
CSRTSNE = csr_mod.CSR_TSNE

# ─── 3) Choose a small set of (λ,γ) pairs to test ─────────────────────
to_try = [
    (0.0, 0.0),
    (0.0, 0.01),
    (0.0, 0.05),
    (1.0, 0.0),
    (1.0, 0.01),
    (1.0, 0.05),
]

# ─── 4) Loop over those pairs, using fewer iterations and fewer samples ─
dataset_name = "s_curve"   # or "swiss_roll"
X, labels = load_small_dataset(dataset_name, n_samples=500)

for lam, gamma in to_try:
    print(f"\n>>> Running CSR-tSNE on {dataset_name} with λ={lam}, γ={gamma}")
    model = CSRTSNE(
        n_neighbors=10,
        lam=lam,
        gamma=gamma,
        random_state=42,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate=200.0,
        n_iter=200,           # fewer iterations for speed
        metric="euclidean",
        n_jobs=1,
        verbose=False
    )
    Y = model.fit(X)

    # Plot immediately
    plt.figure(figsize=(4, 3))
    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="Spectral", s=8, linewidths=0)
    plt.title(f"{dataset_name}  λ={lam}, γ={gamma}")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()
