import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PCA:
    """PCA: center -> covariance -> eigendecomposition -> select top k."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # 1. Center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2. Covariance matrix (unbiased, divide by n-1)
        # because it's mean-centered, entry (i,j) of cov is E[X_iX_j] so just take matrix
        # product with self and divide by (n-1) to get sample estimate of E[X_iX_j]
        cov = (Xc.T @ Xc) / (n - 1)  # Xc_T * Xc

        # 3. Eigendecomposition. cov is symmetric, so eigh is appropriate
        #    (real eigenvalues, orthonormal eigenvectors, ascending order).
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 4. Sort descending, keep top k
        order = np.argsort(eigvals)[::-1] # order is array of indices sorted by corresp values
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # Sign convention: make the largest-magnitude entry of each component positive
        # (eigenvector signs are arbitrary).
        for j in range(eigvecs.shape[1]):
            if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
                eigvecs[:, j] *= -1

        self.cov_ = cov
        self.all_eigenvalues_ = eigvals
        self.components_ = eigvecs[:, : self.n_components].T      # (k, d)
        self.explained_variance_ = eigvals[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigvals.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    # Same data as before: 2D correlated Gaussian, sigma = (1, 2), rho = 0.8
    rng = np.random.default_rng(69)
    s1, s2, rho = 1.0, 2.0, 0.8
    cov_true = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    X = rng.multivariate_normal([0, 0], cov_true, size=1000)

    pca = PCA(n_components=2).fit(X)
    print("Mean:", pca.mean_)
    print("Covariance:\n", pca.cov_)
    print("Eigenvalues (desc):", pca.all_eigenvalues_)
    print("Components (rows):\n", pca.components_)
    #print("Explained variance ratio:", pca.explained_variance_ratio_)

    # k = 1 projection
    pca1 = PCA(n_components=1).fit(X)
    Xr = pca1.inverse_transform(pca1.transform(X))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.35, color="tab:blue", label="data")
    for i, (c, v) in enumerate(zip(pca.components_, pca.all_eigenvalues_)):
        length = 2 * np.sqrt(v)  # 2 standard deviations along the component
        ax.annotate("", xy=pca.mean_ + length * c, xytext=pca.mean_,
                    arrowprops=dict(arrowstyle="->", lw=2.5,
                                    color="tab:red" if i == 0 else "tab:green"))
        ax.plot([], [], color="tab:red" if i == 0 else "tab:green", lw=2.5,
            label=f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})")
    ax.scatter(Xr[:, 0], Xr[:, 1], s=6, color="black", alpha=0.4, label="rank-1 projection")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_title("PCA")
    ax.grid(alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("pca.png", dpi=150)
