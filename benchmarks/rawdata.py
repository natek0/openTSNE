#!/usr/bin/env python3
"""
rawdata.py

Provides functions to generate (or load) standard manifold datasets:
- S-curve  (3D)
- Swiss roll (3D)

Usage:
    from rawdata import get_s_curve, get_swiss_roll
    X_s, col_s = get_s_curve()
    X_w, col_w = get_swiss_roll()
"""

import numpy as np
from sklearn import datasets


def get_s_curve(n_samples=2000, noise=0.05, random_state=42):
    """
    Generate a 3D S-curve dataset.

    Parameters
    ----------
    n_samples : int
        Number of points to sample (default: 2000).
    noise : float
        Standard deviation of Gaussian noise added to the data (default: 0.05).
    random_state : int
        Random seed (default: 42).

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The 3D coordinates of the S-curve.
    color : ndarray of shape (n_samples,)
        The univariate “color” or parameter for each point (in [0, 1]) used to
        color-code the manifold structure.
    """
    X, color = datasets.make_s_curve(
        n_samples=n_samples, noise=noise, random_state=random_state
    )
    return X, color


def get_swiss_roll(n_samples=2000, noise=0.05, random_state=42):
    """
    Generate a 3D Swiss-roll dataset.

    Parameters
    ----------
    n_samples : int
        Number of points to sample (default: 2000).
    noise : float
        Standard deviation of Gaussian noise added to the data (default: 0.05).
    random_state : int
        Random seed (default: 42).

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The 3D coordinates of the Swiss-roll.
    color : ndarray of shape (n_samples,)
        The univariate “color” or parameter for each point (in [0, 1]) used to
        color-code the manifold structure.
    """
    X, color = datasets.make_swiss_roll(
        n_samples=n_samples, noise=noise, random_state=random_state
    )
    return X, color


# If you ever want to still produce a PNG of the raw Swiss-roll for reference,
# you can run this file as a script (e.g. `python rawdata.py`). That code is
# below, but it will not run on import.

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # By default, generate Swiss roll and save a 3D scatter in ./figs_raw
    X_w, col_w = get_swiss_roll()
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_w[:, 0], X_w[:, 1], X_w[:, 2], c=col_w, cmap="Spectral", s=5)
    ax.set_title("Raw Swiss Roll (3D)")
    ax.grid(True)
    ax.view_init(elev=30, azim=-45)

    os.makedirs("figs_raw", exist_ok=True)
    outpath = os.path.join("figs_raw", "swiss_roll_raw.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure ➜ {outpath}")
