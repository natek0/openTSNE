# benchmarks/csr_callbacks.py

import numpy as np

__all__ = ["_ensure_buffer"]

def _ensure_buffer(embedding):
    """
    Ensure that `embedding._extra_forces` exists and has shape (n_samples, 2).
    If not, create it (zeros). Return that array.
    """
    if not hasattr(embedding, "_extra_forces"):
        Y = embedding.view(np.ndarray)    # shape: (n_samples, 2)
        n_samples, dim = Y.shape
        embedding._extra_forces = np.zeros((n_samples, dim), dtype="float64")
    return embedding._extra_forces
