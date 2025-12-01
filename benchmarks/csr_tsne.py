# benchmarks/csr_tsne.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from openTSNE import TSNE
from openTSNE.tsne import TSNEEmbedding
from scipy.sparse import coo_matrix, diags, eye


class CSRTSNE(TSNE):
    """
    A CSR‐penalized t-SNE: standard t-SNE plus two penalties on the 2D embedding:
      1) Spectral penalty: encourages the normalized Laplacian of Y to match that of X.
      2) Curvature penalty: encourages the trace of each local 2D covariance in Y
         to match the trace of its corresponding high-D covariance in X.

    Provides fit(X) and fit_transform(X) for use in benchmark.py.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        lam: float = 0.0,
        gamma: float = 0.0,
        random_state: int = None,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        metric: str = "euclidean",
        n_jobs: int = 1,
        verbose: bool = False,
        **tsne_kwargs,
    ):
        """
        Parameters
        ----------
        n_components : int
            Dimension of the embedding (usually 2).
        n_neighbors : int
            Number of neighbors used for both spectral‐ and curvature‐penalties.
        lam : float
            Weight of the spectral penalty.
        gamma : float
            Weight of the curvature penalty.
        random_state : int or None
            Seed for t-SNE randomness.
        perplexity : float
            Perplexity parameter for t-SNE.
        early_exaggeration : float
            Early exaggeration factor for t-SNE.
        learning_rate : float
            Learning rate for t-SNE optimization.
        n_iter : int
            Number of iterations for t-SNE.
        metric : str
            Distance metric for t-SNE.
        n_jobs : int
            Number of parallel jobs for k-NN computations.
        verbose : bool
            If True, print progress messages.
        tsne_kwargs : dict
            Any additional keywords accepted by openTSNE.TSNE.
        """
        # Store CSR hyperparameters
        self.n_neighbors = n_neighbors
        self.lam = lam
        self.gamma = gamma

        # Keep the learning_rate for later use in apply_csr()
        self._csr_learning_rate = learning_rate

        # Call parent TSNE constructor, omitting unsupported keywords like init/method/angle
        super().__init__(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            # Provide a dummy callback; we’ll override this in fit()
            callbacks=[lambda iteration, error, embedding: None],
            **tsne_kwargs,
        )

        # Placeholders for high-dimensional graph data after calling fit(...)
        self._H_X_list = None   # list of high-D covariances (shape D×D) for each point’s X‐neighbors
        self._L_X = None        # normalized Laplacian of the X‐graph (sparse CSR, shape n×n)
        self._sigma = None      # (optional) bandwidth on X (used in spectral penalty)

    def fit(self, X: np.ndarray, y=None):
        """
        1) Build a k‐NN graph on X to compute:
             - A high-D covariance matrix H_X_list[i] for each point i.
             - The sparse normalized Laplacian L_X of that graph.
        2) Define two callbacks:
             a) csr_grad(...) to compute and accumulate penalty gradients
                into embedding._extra_forces.
             b) apply_csr(...) to subtract (learning_rate * _extra_forces) from Y, clamp Y, and zero it.
        3) Call super().fit(X) so that each optimizer step runs:
             - KL gradient update (openTSNE)
             - csr_grad  → accumulate extra forces
             - apply_csr → apply and clamp extra forces
        """
        n_samples, _ = X.shape
        k = self.n_neighbors

        # --- (1A) Build k-NN graph on X to compute local covariances H_X_list ---
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
        distances, indices = nbrs.kneighbors(X)

        self._H_X_list = []
        for i in range(n_samples):
            Xi = X[indices[i]]           # shape (k, D)
            cov_i = np.cov(Xi.T)         # shape (D, D)
            self._H_X_list.append(cov_i)

        # --- (1B) Build sparse normalized Laplacian L_X for X-graph ---
        rows = []
        cols = []
        data = []
        for i in range(n_samples):
            for j in indices[i]:
                rows.append(i); cols.append(j); data.append(1.0)
                rows.append(j); cols.append(i); data.append(1.0)
        A = coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        deg = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
        D_inv_sqrt = diags(d_inv_sqrt)
        I_sparse = eye(n_samples, format="csr")
        self._L_X = I_sparse - (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

        # Store a global sigma based on X (use average k-th neighbor distance)
        self._sigma = np.mean(distances[:, -1])

        # --- (2A) Define csr_grad callback with nonzero spectral gradient ---
        def csr_grad(iteration, error, embedding: TSNEEmbedding):
            """
            Called each optimizer step. Accumulates spectral and curvature penalty gradients
            into embedding._extra_forces (shape n×2). Does NOT modify the KL gradient.
            """
            Y = embedding.view(np.ndarray)         # shape (n_samples, 2)

            # Step (A): ensure the “extra forces” buffer exists
            from benchmarks.csr_callbacks import _ensure_buffer
            buf = _ensure_buffer(embedding)         # shape (n_samples, 2)

            sigma = self._sigma
            n = Y.shape[0]
            eps = 1e-8

            # 1) Build k-NN graph on Y
            nbrs_Y = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(Y)
            _, indices_Y = nbrs_Y.kneighbors(Y)

            # 2) Precompute Gaussian weights w[i,j] for neighbors
            w_ij = {}
            for i in range(n):
                for j in indices_Y[i]:
                    if i == j:
                        continue
                    diff = Y[i] - Y[j]   # shape (2,)
                    wij = np.exp(-np.dot(diff, diff) / (2 * sigma**2))
                    w_ij[(i, j)] = wij
                    w_ij[(j, i)] = wij

            # 3) Compute degrees d[i] = sum_{j≠i} w_ij, then clamp to ≥ eps
            d = np.zeros(n, dtype=float)
            for i in range(n):
                total = 0.0
                for j in indices_Y[i]:
                    if i == j:
                        continue
                    total += w_ij.get((i, j), 0.0)
                d[i] = max(total, eps)

            # 4) Define a safe L_Y[i,j] that returns 0 if either d[i] or d[j] is too small
            def Lij(i, j):
                if i == j:
                    return 1.0
                dij = d[i] * d[j]
                if dij < eps:
                    return 0.0
                return -w_ij.get((i, j), 0.0) / np.sqrt(dij)

            # 5) Compute spec_grad[i]
            spec_grad = np.zeros_like(Y)           # shape (n_samples, 2)
            for i in range(n):
                # 5A) Compute ∂d_i/∂Y_i = - (1/σ²) ∑_{j≠i} (Y_i - Y_j) * w_ij
                grad_di = np.zeros(2)
                for j in indices_Y[i]:
                    if i == j:
                        continue
                    grad_di += - (1.0 / (sigma**2)) * (Y[i] - Y[j]) * w_ij.get((i, j), 0.0)

                # 5B) Loop over neighbors to accumulate ∂L_Y[i,j]/∂Y_i
                for j in indices_Y[i]:
                    if i == j:
                        continue

                    denom2 = d[i] * d[j]
                    if denom2 < eps:
                        # skip this pair if either degree is essentially zero
                        continue

                    # ∂w_ij/∂Y_i = - (1/σ²) * (Y_i - Y_j) * w_ij
                    grad_wij = - (1.0 / (sigma**2)) * (Y[i] - Y[j]) * w_ij.get((i, j), 0.0)

                    # Now compute ∂L_Y[i,j]/∂Y_i
                    # term1 = - grad_wij / sqrt(d[i]*d[j])
                    sqrt_prod = np.sqrt(denom2)
                    term1_raw = - grad_wij / sqrt_prod
                    term1 = np.nan_to_num(term1_raw, nan=0.0, posinf=0.0, neginf=0.0)

                    # term2 = w_ij * (1/2) * (d[i]*d[j])^(-3/2) * [d[j]*grad_di + d[i]*grad_wij]
                    wval = w_ij.get((i, j), 0.0)
                    factor_raw = 0.5 * wval / ( (d[i]**1.5) * np.sqrt(d[j]) )
                    factor = np.nan_to_num(factor_raw, nan=0.0, posinf=0.0, neginf=0.0)

                    inner = d[j] * grad_di + d[i] * grad_wij
                    term2_raw = factor * inner
                    term2 = np.nan_to_num(term2_raw, nan=0.0, posinf=0.0, neginf=0.0)

                    dLy_dYi = term1 + term2

                    Lyij = Lij(i, j)
                    Lxij = self._L_X[i, j]
                    diff_L = Lyij - Lxij
                    spec_grad[i] += 2.0 * diff_L * dLy_dYi

            # 6) Compute curvature penalty (via traces)
            curv_grad = np.zeros_like(Y)          # shape (n_samples, 2)
            for i in range(n):
                nbr_idx = indices_Y[i]
                Yi = Y[nbr_idx]                   # shape (k, 2)
                cov_Yi = np.cov(Yi.T)             # shape (2, 2)
                trace_Yi = np.trace(cov_Yi)       # scalar
                trace_Hi = np.trace(self._H_X_list[i])  # high-D trace
                diff_trace = trace_Yi - trace_Hi
                mean_Yi = Yi.mean(axis=0)         # shape (2,)
                # ∂ trace(cov_Yi) / ∂ Y[i] = (2/k) * (Y[i] - mean_Yi)
                dTrace_dYi = (2.0 / float(len(nbr_idx))) * (Y[i] - mean_Yi)
                curv_grad[i] = 2.0 * diff_trace * dTrace_dYi

            # 7) Clamp NaN/Inf in both gradients
            spec_grad = np.nan_to_num(spec_grad, nan=0.0, posinf=0.0, neginf=0.0)
            curv_grad = np.nan_to_num(curv_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # 8) Accumulate into “extra forces” buffer
            buf[:] += 2.0 * self.lam   * spec_grad
            buf[:] += 2.0 * self.gamma * curv_grad

        # --- (2B) Define apply_csr callback (applies stored extra forces) ---
        def apply_csr(iteration, error, embedding: TSNEEmbedding):
            buf = getattr(embedding, "_extra_forces", None)
            if buf is None:
                return
            Y = embedding.view(np.ndarray)         # shape (n_samples, 2)

            # Subtract (learning_rate × buf) from Y:
            Y[:] -= self._csr_learning_rate * buf

            # Clamp any NaN/Inf in Y:
            Y[:] = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip extremely large values to ±1e3:
            np.clip(Y, -1e3, 1e3, out=Y)

            # Zero out the buffer for the next iteration
            buf[:] = 0.0

        # --- (2C) Register callbacks in order ---
        self.callbacks = [
            csr_grad,   # compute & accumulate penalties into _extra_forces
            apply_csr   # apply & clamp extra forces, then zero the buffer
        ]

        # --- (3) Run the standard TSNE.fit with our callbacks enabled ---
        Y = super().fit(X, y)
        return Y

    def fit_transform(self, X: np.ndarray, y=None):
        """
        So benchmark.py will call fit_transform(X) instead of transform(X).
        """
        return self.fit(X, y)


# Alias so that benchmark.py can import CSR_TSNE by name
CSR_TSNE = CSRTSNE
