import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(69)

# Parameters
n = 1000
mean = np.array([0.0, 0.0])
sigma = np.array([1.0, 2.0])          # std devs of x1, x2

# generate a 2D data set of two Gaussians with target correlation rho
def make_data(rho, n=n):
    cov = np.array([
        [sigma[0] ** 2,             rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ])
    return rng.multivariate_normal(mean, cov, size=n), cov

# single dataset (rho = 0.8) with marginal-free scatter
X, cov = make_data(0.8)
print("Target covariance: \n", cov)
print("Sample covariance: \n", np.cov(X.T))
print("Sample correlation: ", np.corrcoef(X.T)[0, 1])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
ax.set_xlabel("x_1")
ax.set_ylabel("x_2")
ax.set_title("Correlated Gaussians")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.5)
fig.tight_layout()
fig.savefig("scatter_single.png", dpi=150)
