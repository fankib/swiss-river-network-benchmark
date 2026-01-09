import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

# --- Objective function (black-box) ---
# This function accepts scalars, 1D arrays or 2D column arrays (n,1).
def objective(x):
    x = np.asarray(x)
    return -(np.sin(3*x) + x**2 - 0.7*x)

# --- Expected Improvement function ---
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Compute the Expected Improvement at points X based on existing samples
    X: (n_candidates, 1)
    X_sample: (n_samples, 1)
    Y_sample: (n_samples, 1) or (n_samples,)
    gpr: fitted GaussianProcessRegressor
    """
    mu, sigma = gpr.predict(X, return_std=True)
    # Y_sample may be (n,1); take max over all entries
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def propose_location(gpr, X_sample, Y_sample, bounds, n_candidates=1000):
    """Propose the next sampling point by maximizing EI over a grid."""
    X = np.linspace(bounds[0, 0], bounds[0, 1], n_candidates).reshape(-1, 1)
    ei = expected_improvement(X, X_sample, Y_sample, gpr)
    idx = np.argmax(ei)
    # Return a 2D array shape (1,1) so stacking is consistent
    return X[idx].reshape(1, 1), ei


def run_bo(n_init=5, n_iter=10, random_seed=42, plot=True):
    np.random.seed(random_seed)

    # Bounds (1D problem)
    bounds = np.array([[-2.0, 2.0]])

    # Initial samples (n_init x 1)
    X_init = np.random.uniform(bounds[0, 0], bounds[0, 1], size=(n_init, 1))
    Y_init = objective(X_init).reshape(-1, 1)

    # Gaussian Process with Matern kernel
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    X_sample = X_init.copy()
    Y_sample = Y_init.copy()

    for i in range(n_iter):
        # Fit GP (y must be 1d for sklearn's fit)
        gpr.fit(X_sample, Y_sample.ravel())

        # Propose next point
        X_next, ei = propose_location(gpr, X_sample, Y_sample, bounds)

        # Evaluate objective at X_next. Keep shape (1,1).
        Y_next = objective(X_next).reshape(1, 1)

        # --- FIXED: ensure shapes are consistent before stacking ---
        # X_sample: (n_samples, 1), X_next: (1,1) -> vstack OK
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

        print(f"Iteration {i+1}: x = {X_next.item():.6f}, f(x) = {Y_next.item():.6f}")

    # Final fit
    gpr.fit(X_sample, Y_sample.ravel())

    if plot:
        X = np.linspace(bounds[0, 0], bounds[0, 1], 1000).reshape(-1, 1)
        mu, sigma = gpr.predict(X, return_std=True)

        plt.figure(figsize=(10, 6))
        plt.plot(X, objective(X), 'y--', label='True function')
        plt.plot(X, mu, 'b-', label='GP mean')
        plt.fill_between(X.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma, alpha=0.2)
        plt.scatter(X_sample, Y_sample, c='red', s=40, label='Samples')
        plt.legend()
        plt.title("Bayesian Optimization with GPR + Expected Improvement")
        plt.show()

    return X_sample, Y_sample, gpr


# -------------------- simple tests --------------------

def _test_shapes():
    """Run a short BO (no plotting) and verify sample arrays shapes."""
    X_sample, Y_sample, _ = run_bo(n_init=3, n_iter=2, plot=False)
    assert X_sample.ndim == 2 and X_sample.shape[1] == 1, "X_sample should be (n_samples,1)"
    assert Y_sample.ndim == 2 and Y_sample.shape[1] == 1, "Y_sample should be (n_samples,1)"
    print("Shape tests passed.")


def _test_ei_length():
    """Quick unit test: EI should match number of candidate points."""
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    X_sample = np.array([[-1.0], [0.5]])
    Y_sample = objective(X_sample).reshape(-1, 1)
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
    gpr.fit(X_sample, Y_sample.ravel())
    ei = expected_improvement(X, X_sample, Y_sample, gpr)
    assert ei.shape[0] == X.shape[0], "EI must have same length as candidate X"
    print("EI length test passed.")


if __name__ == "__main__":
    # Run small tests first
    _test_shapes()
    _test_ei_length()

    # Run a longer BO and show plots
    run_bo(n_init=5, n_iter=8, plot=True)

