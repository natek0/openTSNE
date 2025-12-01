import numpy as np
import warnings
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from openTSNE import TSNE, TSNEEmbedding

def _ensure_buffer(embedding: TSNEEmbedding):
    """
    Ensure that embedding._extra_forces exists and return it.
    """
    buf = getattr(embedding, "_extra_forces", None)
    if buf is None:
        # Create a buffer of same shape as embedding
        buf = np.zeros_like(embedding.view(np.ndarray))
        embedding._extra_forces = buf
    return buf


class AdaptiveEigenTSNE(TSNE):
    """
    t-SNE variant with adaptive eigenvalue-based regularization (AE-TSNE).
    Implements analytic eigenvalue gradients per the proposal.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        lam_init: float = 0.1,
        lam_max: float = 2.0,
        n_eigvals: int = 8,
        update_freq: int = 20,
        rho: float = 0.1,
        eta_lambda: float = 0.02,
        beta_m: float = 0.9,
        beta_anneal: float = 0.01,
        alpha: float = 1e-8,
        epsilon: float = 1e-6,
        n_iter: int = 1000,
        learning_rate: float = 200.0,
        metric: str = "euclidean",
        spectral_norm: str = "gap",  # one of {"gap", "log", "l2"}
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize AE-TSNE with adaptive eigenvalue penalty.

        Parameters
        ----------
        n_components : int
            Dimensionality of the embedding space (default: 2).
        n_neighbors : int
            Number of nearest neighbors for Laplacian construction (default: 15).
        lam_init : float
            Initial penalty weight lambda (default: 0.1).
        lam_max : float
            Maximum penalty weight lambda (default: 2.0).
        n_eigvals : int
            Number of eigenvalues (excluding trivial) to match (default: 8).
        update_freq : int
            Frequency (in iterations) to update eigen penalty (default: 20).
        rho : float
            Target gradient ratio (default: 0.1).
        eta_lambda : float
            Learning rate for lambda adaptation (default: 0.02).
        beta_m : float
            Momentum coefficient for lambda updates (default: 0.9).
        beta_anneal : float
            Annealing rate for base lambda schedule (default: 0.01).
        alpha : float
            Regularization weight on eigenvalues (default: 1e-8).
        epsilon : float
            Small constant for numerical stability (default: 1e-6).
        n_iter : int
            Total number of iterations for t-SNE (default: 1000).
        learning_rate : float
            Learning rate for t-SNE (default: 200.0).
        metric : str
            Distance metric (passed to openTSNE) (default: "euclidean").
        spectral_norm : str
            Which spectral normalization to use ("gap", "log", or "l2").
        verbose : bool
            If True, print warnings and debug info.
        **kwargs : dict
            Additional keyword arguments for openTSNE.TSNE.
        """
        super().__init__(
            n_components=n_components,
            perplexity=30,
            metric=metric,
            n_iter=n_iter,
            learning_rate=learning_rate,
            callbacks=[],  # we'll set them in fit()
            **kwargs
        )
        self.n_neighbors = n_neighbors
        self.lam_init = lam_init
        self.lam_max = lam_max
        self.n_eigvals = n_eigvals
        self.update_freq = update_freq
        self.rho = rho
        self.eta_lambda = eta_lambda
        self.beta_m = beta_m
        self.beta_anneal = beta_anneal
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.spectral_norm = spectral_norm
        self.verbose = verbose

        # Internal state
        self._ref_spec = None
        self._lam = lam_init
        self._momentum = 0.0
        self._last_eig_update = 0
        self._last_kl_norm = 1.0

    def _build_regularized_laplacian(self, X: np.ndarray):
        """
        Build normalized + regularized Laplacian L_reg = (I – D^{-1/2} W D^{-1/2}) + εI.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_components)
            Current embedding or input data.

        Returns
        -------
        L_reg : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
            Regularized normalized Laplacian.
        """
        n_samples, _ = X.shape
        k = min(self.n_neighbors, n_samples - 1)

        # 1) k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
        distances, indices = nbrs.kneighbors(X)

        # 2) Bandwidth σ = median of distances to 2nd nearest neighbor + ε
        sigma = np.median(distances[:, 1:]) + self.epsilon

        rows, cols, data = [], [], []
        for i in range(n_samples):
            for jidx, j in enumerate(indices[i]):
                if i == j:
                    continue
                dist = distances[i, jidx]
                w = np.exp(-dist**2 / (2 * sigma**2))
                rows.append(i); cols.append(j); data.append(w)
                rows.append(j); cols.append(i); data.append(w)

        A = coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples)).tocsr()

        # 3) If graph has >1 component, add ε-weight weak link between closest points.
        from scipy.sparse.csgraph import connected_components
        n_comp, labels = connected_components(A, directed=False)
        if n_comp > 1 and self.verbose:
            print(f"[AE-TSNE] Graph has {n_comp} components; adding weak links.")
            for comp_i in range(n_comp):
                for comp_j in range(comp_i + 1, n_comp):
                    pts_i = np.where(labels == comp_i)[0]
                    pts_j = np.where(labels == comp_j)[0]
                    # Find closest pair
                    distmat = np.linalg.norm(X[pts_i][:, None, :] - X[pts_j][None, :, :], axis=2)
                    idx = np.unravel_index(np.argmin(distmat), distmat.shape)
                    i0 = pts_i[idx[0]]
                    j0 = pts_j[idx[1]]
                    A[i0, j0] = self.epsilon
                    A[j0, i0] = self.epsilon

        # 4) Form normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(deg + self.epsilon)
        D_inv_sqrt = diags(deg_inv_sqrt)
        I = eye(n_samples, format="csr")
        L = I - (D_inv_sqrt @ A @ D_inv_sqrt)

        # 5) Add εI
        return (L + self.epsilon * I).tocsr()

    def _compute_eigendecomposition(self, L):
        """
        Compute the (m+1) smallest nonzero eigenvalues and eigenvectors of L_reg.

        Returns
        -------
        eigvals : ndarray, shape (m+1,)
            Smallest nonzero eigenvalues, sorted ascending.
        eigvecs : ndarray, shape (n_samples, m+1)
            Corresponding eigenvectors. If eigendecomposition fails, returns (eigvals, None).
        """
        n = L.shape[0]
        m = self.n_eigvals
        try:
            # 1) Try shift-invert to get m+1 smallest nonzero eigenvalues
            kcalc = min(m + 2, n - 2)
            vals, vecs = eigsh(L, k=kcalc, which="LM", sigma=0.01, maxiter=2000)
            realvals = np.real(vals)
            sorted_idxs = np.argsort(realvals)
            realvals = realvals[sorted_idxs]
            # Filter out any that are ~0 or negative
            realvals = realvals[realvals > self.epsilon]
            if realvals.shape[0] < m + 1:
                raise RuntimeError("Too few eigenvalues from shift-invert")
            # Keep the first m+1
            realvals = realvals[: m + 1]

            # Recompute eigenvectors exactly for those indices using SM
            vals_sm, vecs_sm = eigsh(L, k=m + 1, which="SM", maxiter=5000)
            idx_sm = np.argsort(np.real(vals_sm))
            eigvals = np.real(vals_sm[idx_sm])[: m + 1]
            eigvecs = vecs_sm[:, idx_sm[: m + 1]]

        except Exception:
            # Fallback #1: SM directly
            try:
                vals_sm, vecs_sm = eigsh(L, k=m + 1, which="SM", maxiter=5000)
                idx_sm = np.argsort(np.real(vals_sm))
                eigvals = np.real(vals_sm[idx_sm])[: m + 1]
                eigvecs = vecs_sm[:, idx_sm[: m + 1]]
            except Exception:
                # Fallback #2: dense (if n <= 100)
                if n <= 100:
                    full_vals, full_vecs = np.linalg.eig(L.toarray())
                    idx_all = np.argsort(np.real(full_vals))
                    eigvals = np.real(full_vals[idx_all])[: m + 1]
                    eigvecs = full_vecs[:, idx_all[: m + 1]]
                else:
                    # Fallback #3: synthetic (no meaningful eigenvectors)
                    eigvals = np.linspace(0.1, 1.0, m + 1)
                    eigvecs = None

        return eigvals, eigvecs

    def _normalize_eigenvalues(self, eigvals: np.ndarray) -> np.ndarray:
        """
        Normalize raw eigenvalues according to self.spectral_norm.

        Returns
        -------
        spec : ndarray, shape (n_eigvals,)
            Normalized eigen-spectrum (length m).
        """
        m = self.n_eigvals
        if self.spectral_norm == "gap":
            # Compute differences λ2-λ1, λ3-λ2, ..., λ_{m+1}-λ_m
            gaps = eigvals[1:] - eigvals[:-1]
            # If gaps < m (rare), pad with last gap
            if gaps.shape[0] < m:
                pad = np.ones(m - gaps.shape[0]) * gaps[-1]
                gaps = np.concatenate([gaps, pad])
            return gaps[:m] / (np.mean(eigvals) + self.epsilon)

        elif self.spectral_norm == "log":
            logv = np.log(eigvals + self.epsilon)
            normed = logv / (np.std(logv) + self.epsilon)
            # Return first m entries (skip the trivial λ0, if needed)
            return normed[1 : m + 1]

        elif self.spectral_norm == "l2":
            vec = eigvals.copy()
            if vec.shape[0] < m:
                # If fewer than m+1 eigenvalues, pad
                pad = np.ones(m + 1 - vec.shape[0]) * vec[-1]
                vec = np.concatenate([vec, pad])
            vec = vec[: m]  # take first m
            den = max(np.linalg.norm(vec), self.epsilon)
            return vec / den

        else:
            raise ValueError(f"Unsupported spectral_norm = {self.spectral_norm}")

    def _group_degenerate_indices(self, eigvals: np.ndarray, tol: float = 1e-8):
        """
        Identify groups of near-degenerate eigenvalues.

        Returns
        -------
        groups : list of lists
            Each sublist contains indices of eigvals that are within tol of each other.
        """
        n = len(eigvals)
        processed = set()
        groups = []
        for i in range(n):
            if i in processed:
                continue
            group = [i]
            processed.add(i)
            for j in range(i + 1, n):
                if abs(eigvals[i] - eigvals[j]) < tol:
                    group.append(j)
                    processed.add(j)
            groups.append(group)
        return groups

    def _compute_laplacian_derivative(self, Y: np.ndarray, eigvec: np.ndarray) -> np.ndarray:
        """
        Compute ∂(vᵀ L_reg v)/∂Y for a given eigenvector v.

        Uses first-order perturbation: ∂λ / ∂Y_{i,d} = vᵀ (∂L_reg/∂Y_{i,d}) v.

        Returns
        -------
        dL_dY : ndarray, shape (n_samples, dim)
            Gradient of that eigenvalue w.r.t. each coordinate of Y.
        """
        n, dim = Y.shape
        # 1) Build kNN graph & compute σ
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(Y)
        distances, indices = nbrs.kneighbors(Y)
        sigma = np.median(distances[:, 1:]) + self.epsilon

        # 2) Build adjacency matrix A and degree
        rows, cols, data = [], [], []
        for i in range(n):
            for jidx, j in enumerate(indices[i]):
                if i == j:
                    continue
                diff = Y[i] - Y[j]
                w = np.exp(-np.sum(diff**2) / (2 * sigma**2))
                rows.append(i); cols.append(j); data.append(w)
                rows.append(j); cols.append(i); data.append(w)
        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(deg + self.epsilon)

        # 3) Compute ∂L / ∂Y: L = I - D^{-1/2} A D^{-1/2}
        # We approximate derivative by focusing on dW/dY and ignoring second-order deg-term derivatives.
        dL_dY = np.zeros_like(Y)

        for i in range(n):
            inv_sqrt_i = deg_inv_sqrt[i]
            for jidx, j in enumerate(indices[i]):
                if i == j:
                    continue
                # Weight and its derivative wrt Y[i]:
                diff = Y[i] - Y[j]
                w = np.exp(-np.sum(diff**2) / (2 * sigma**2))
                # ∂W_{ij}/∂Y_i = (diff/σ²)*w
                dW_ij = (diff / (sigma**2)) * w

                inv_sqrt_j = deg_inv_sqrt[j]
                # Contribution to ∂(D^{-1/2} A D^{-1/2})_{i,j} wrt Y[i]:
                # ≈ inv_sqrt_i * dW_ij * inv_sqrt_j
                # And since L = I - that, ∂L_{ij}/∂Y_i = - (inv_sqrt_i * dW_ij * inv_sqrt_j)
                contrib_ij = -inv_sqrt_i * dW_ij * inv_sqrt_j

                # The eigenvalue perturbation dλ = Σ_{p,q} v[p] * (∂L_{p,q}/∂Y_i) * v[q].
                # Nonzero ∂L_{p,q}/∂Y_i arise only for (p,q) = (i,j) or (j,i) or diagonal corrections.
                # For simplicity, we include only (i,j) and (j,i):
                #   ∂L_{j,i}/∂Y_i = -inv_sqrt_j * dW_ij * inv_sqrt_i
                contrib_ji = -inv_sqrt_j * dW_ij * inv_sqrt_i

                # Now accumulate gradient: dλ/∂Y_i = v[i]*(contrib_ij)*v[j] + v[j]*(contrib_ji)*v[i]
                # = 2 * v[i]*v[j] * contrib_ij    (since contrib_ij = contrib_ji)
                dL_ij = 2.0 * eigvec[i] * eigvec[j] * contrib_ij

                # Add to row i of dL_dY
                dL_dY[i] += eigvec[i] * dL_ij

        return dL_dY

    def _compute_analytical_eigen_gradient(
        self,
        Y: np.ndarray,
        eigvals_Y: np.ndarray,
        eigvecs_Y: np.ndarray,
        spec_diff: np.ndarray
    ) -> np.ndarray:
        """
        Analytical gradient of the eigenvalue-based loss w.r.t. Y.

        Loss = ||spec_Y – ref_spec||^2 + α ||spec_Y||^2.

        ∂loss/∂Y = Σ_k [ 2 (spec_Y[k] – ref_spec[k]) + 2 α spec_Y[k] ] * ∂spec_Y[k]/∂Y.

        Returns
        -------
        g_eig : ndarray, shape (n_samples, dim)
            Gradient w.r.t. each coordinate of Y.
        """
        n, dim = Y.shape
        m = self.n_eigvals
        spec_Y = spec_diff + self._ref_spec 
        g_eig = np.zeros_like(Y)

        # Handle degenerate eigenvalues by grouping
        groups = self._group_degenerate_indices(eigvals_Y[: m + 1], tol=1e-8)

        # For each eigenvalue index k in 0..m-1 (skip trivial 0-th if desired)
        for group in groups:
            # Only process indices in [1..m] (skip index 0, the trivial eigen)
            filtered = [k for k in group if 1 <= k <= m]
            if not filtered:
                continue

            # For each eigen‐index k in this group:
            for k in filtered:
                # The corresponding index in spec_Y is k-1
                idx_spec = k - 1

                # Eigenvector associated with eigenvalue eigvals_Y[k]
                v_k = eigvecs_Y[:, k]

                # Compute ∂λ_k / ∂Y via perturbation: ∂L/∂Y projected onto v_k
                dL_dY_k = self._compute_laplacian_derivative(Y, v_k)

                # Coefficient: 2*( (spec_Y[k-1] - ref_spec[k-1]) + α * spec_Y[k-1] )
                # But spec_diff[k-1] = spec_Y[k-1] - ref_spec[k-1]
                coef = 2.0 * (spec_diff[idx_spec] + self.alpha * spec_Y[idx_spec])

                # Accumulate gradient
                g_eig += coef * dL_dY_k

        return g_eig

    def _detect_numerical_instability(self, Y: np.ndarray, spec_diff: np.ndarray) -> bool:
        """
        Check for NaN/Inf, explosive spectral difference, or embedding collapse.

        Returns
        -------
        unstable : bool
            True if numerical instability is detected.
        """
        # 1) NaN or Inf in embedding
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            return True

        # 2) Spectral difference too large
        if np.linalg.norm(spec_diff) > 50.0:
            return True

        # 3) Embedding collapse: any two points < tol apart
        pairwise = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2) + np.eye(Y.shape[0]) * 1e6
        min_dist = np.min(pairwise)
        if min_dist < 1e-6:
            return True

        return False

    def _stabilize_gradient(self, g_raw: np.ndarray, spec_diff: np.ndarray) -> np.ndarray:
        """
        Spectral-error-aware clipping + NaN/Inf filtering.

        Returns
        -------
        g_stable : ndarray, shape same as g_raw
            Possibly clipped gradient.
        """
        C_max = 0.1
        spec_err_norm = np.linalg.norm(spec_diff)
        stabilization_factor = 1.0 / (1.0 + spec_err_norm)
        max_grad_norm = C_max * stabilization_factor

        g_norm = np.linalg.norm(g_raw)
        if g_norm > max_grad_norm:
            g_raw = g_raw * (max_grad_norm / (g_norm + self.epsilon))
            if self.verbose:
                print(f"[AE-TSNE] Gradient clipped: {g_norm:.3e} → {max_grad_norm:.3e}")

        # Check for NaN/Inf
        if np.any(np.isnan(g_raw)) or np.any(np.isinf(g_raw)):
            if self.verbose:
                warnings.warn("[AE-TSNE] Gradient contains NaN/Inf → zeroing out.")
            return np.zeros_like(g_raw)

        return g_raw

    def _setup_enhanced_callbacks(self):
        """
        Install three callbacks in order:
        1) capture_kl_norm: record ||g_KL|| (pure) before eigen forces are added.
        2) eigen_callback: compute eigen gradient, update λ, add to buffer.
        3) apply_forces_callback: apply Y ← Y – lr * buffer, then zero buffer.
        """

        def capture_kl_norm(iteration, error, embedding: TSNEEmbedding):
            if iteration < 250:  # skip early exaggeration
                return
            buf = getattr(embedding, "_extra_forces", None)
            if buf is not None:
                self._last_kl_norm = max(np.linalg.norm(buf), self.epsilon)

        def eigen_callback(iteration, error, embedding: TSNEEmbedding):
            if iteration < 250 or self._lam == 0.0:
                return
            if iteration - self._last_eig_update < self.update_freq:
                return

            self._last_eig_update = iteration
            Y = embedding.view(np.ndarray)
            buf = _ensure_buffer(embedding)

            # Build L_Y and get eigendecomposition
            L_Y = self._build_regularized_laplacian(Y)
            eigvals_Y, eigvecs_Y = self._compute_eigendecomposition(L_Y)
            if eigvecs_Y is None:
                if self.verbose:
                    warnings.warn(f"[AE-TSNE] Eigen decomposition failed at iter {iteration}. Disabling penalty.")
                self._lam = 0.0
                return

            # Normalize eigenvalues to get spec_Y
            spec_Y = self._normalize_eigenvalues(eigvals_Y)
            spec_diff = spec_Y - self._ref_spec

            # Emergency brake if unstable
            if self._detect_numerical_instability(Y, spec_diff):
                if self.verbose:
                    warnings.warn(f"[AE-TSNE] Emergency brake at iter {iteration}. Disabling penalty.")
                self._lam = 0.0
                return

            # Compute analytic eigen gradient (including α term)
            g_eig = self._compute_analytical_eigen_gradient(Y, eigvals_Y, eigvecs_Y, spec_diff)

            # Stabilize/clip gradient
            g_eig = self._stabilize_gradient(g_eig, spec_diff)

            # Adaptive λ update using captured KL norm
            r = np.linalg.norm(g_eig) / (self._last_kl_norm + self.epsilon)
            dlam = self.eta_lambda * (r - self.rho)
            self._momentum = self.beta_m * self._momentum + (1.0 - self.beta_m) * dlam
            lam_base = self.lam_max * (1.0 - np.exp(-self.beta_anneal * iteration / self.n_iter))
            mu = np.clip(1.0 + self._momentum, 0.5, 2.0)
            self._lam = np.clip(lam_base * mu, self.lam_init, self.lam_max)

            # Add eigen-forces to the buffer
            buf[:] += self._lam * g_eig

        def apply_forces_callback(iteration, error, embedding: TSNEEmbedding):
            buf = getattr(embedding, "_extra_forces", None)
            if buf is None:
                return
            Y = embedding.view(np.ndarray)
            Y[:] -= self.learning_rate * buf
            np.clip(Y, -1e3, 1e3, out=Y)
            buf[:] = 0.0

        return [capture_kl_norm, eigen_callback, apply_forces_callback]

    def fit(self, X: np.ndarray, y=None):
        """
        Compute AE-TSNE embedding for X, with adaptive eigenvalue penalty.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input high-dimensional data.
        y : Ignored (for compatibility).

        Returns
        -------
        embedding : TSNEEmbedding
            The low-dimensional embedding.
        """
        # 1) Build reference spectrum from input X
        L_X = self._build_regularized_laplacian(X)
        eigvals_X, _ = self._compute_eigendecomposition(L_X)
        self._ref_spec = self._normalize_eigenvalues(eigvals_X)

        # 2) Reset state
        self._lam = self.lam_init
        self._momentum = 0.0
        self._last_eig_update = 0
        self._last_kl_norm = 1.0

        # 3) Setup callbacks
        self.callbacks = self._setup_enhanced_callbacks()

        # 4) Run parent TSNE
        return super().fit(X, y)
