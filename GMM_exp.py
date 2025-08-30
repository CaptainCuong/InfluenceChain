import numpy as np
from sklearn.mixture import GaussianMixture

def sample_from_gaussians(mu1, sigma1, mu2, sigma2, n1=300, n2=700, random_state=None):
    """
    Generate samples from two Gaussian distributions.

    Parameters:
        mu1, sigma1 : float
            Mean and std. deviation of the first Gaussian (G1).
        mu2, sigma2 : float
            Mean and std. deviation of the second Gaussian (G2).
        n1, n2 : int
            Number of samples from G1 and G2.
        random_state : int or None
            Random seed for reproducibility.

    Returns:
        np.ndarray
            Combined dataset of shape (n1+n2,).
    """
    rng = np.random.default_rng(random_state)

    g1_samples = rng.normal(mu1, sigma1, n1)
    g2_samples = rng.normal(mu2, sigma2, n2)

    return np.concatenate([g1_samples, g2_samples])

def train_gmm_on_data(data, n_components=2, random_state=42, covariance_type="full"):
    """
    Fit a Gaussian Mixture Model on 1D or multi-D data.
    
    Parameters
    ----------
    data : array-like, shape (n_samples,) or (n_samples, n_features)
        Your dataset (e.g., the 1000 samples you generated).
    n_components : int
        Number of Gaussian components to fit (2 for G1/G2).
    random_state : int
        Seed for reproducibility.
    covariance_type : {"full","tied","diag","spherical"}
        GMM covariance structure.

    Returns
    -------
    gmm : fitted sklearn.mixture.GaussianMixture
    """
    X = np.asarray(data)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=10,              # multiple inits for stability
        max_iter=500,           # give EM plenty of steps
        tol=1e-5,               # tighter convergence
        random_state=random_state
    ).fit(X)
    return gmm

def negative_log_likelihood(gmm: GaussianMixture, X: np.ndarray) -> float:
    """
    Compute the Negative Log-Likelihood (NLL) of a dataset under a trained GMM.

    Parameters
    ----------
    gmm : GaussianMixture
        A fitted scikit-learn GaussianMixture model.
    X : np.ndarray
        Data array of shape (n_samples, n_features).

    Returns
    -------
    float
        The negative log-likelihood.
    """
    X = np.asarray(X)
    if X.ndim == 1:   # ensure 2D input
        X = X.reshape(-1, 1)
    log_likelihood = gmm.score(X) * X.shape[0]   # score() = avg log-likelihood
    return -log_likelihood

# Example usage:
data = sample_from_gaussians(mu1=0, sigma1=1, mu2=5, sigma2=1.5, n1=300, n2=700, random_state=42)

print("First 10 samples:", data[:10])
print("Mean of samples:", np.mean(data))
print("Total number of samples:", len(data))

gmm = train_gmm_on_data(data, n_components=2)

print("Weights (mixing proportions):", gmm.weights_)          # ~ [0.3, 0.7] (order may vary)
print("Means:", gmm.means_.ravel())                            # close to [0, 5] (order may vary)
print("Variances:", gmm.covariances_.ravel())                  # ~ [1^2, 1.5^2] if 1D, order may vary

# Soft assignments (responsibilities) for each point:
# resp[i, k] = P(component k | x_i)
resp = gmm.predict_proba(data.reshape(-1, 1))
print("Responsibilities shape:", resp.shape)

# Hard cluster labels if needed:
labels = gmm.predict(data.reshape(-1, 1))
print("Labels distribution:", np.bincount(labels))

# Model quality metrics:
print("Log-likelihood per sample:", gmm.score(data.reshape(-1, 1)))
print("AIC:", gmm.aic(data.reshape(-1, 1)))
print("BIC:", gmm.bic(data.reshape(-1, 1)))

# Sample 1000 new points from the trained GMM (G3)
X_new, y_new = gmm.sample(1000)

print("Sampled data shape:", X_new.shape)
print("First 10 samples:", X_new[:10].ravel())
print("Component labels shape:", y_new.shape)
print("Label counts:", np.bincount(y_new))

# Ensure X_new is 2D (n_samples, n_features)
X_refit = np.asarray(X_new)
if X_refit.ndim == 1:
    X_refit = X_refit.reshape(-1, 1)

# ---- Option A: Fit a 2-component GMM (to mirror G3) ----
gmm_refit = GaussianMixture(
    n_components=2,
    covariance_type="full",
    n_init=10,
    max_iter=500,
    tol=1e-5,
    random_state=7
).fit(X_refit)

print("Refit Weights:", gmm_refit.weights_)
print("Refit Means:", gmm_refit.means_.ravel())
print("Refit Variances:", gmm_refit.covariances_.ravel())  # if 1D
print("Refit AIC:", gmm_refit.aic(X_refit))
print("Refit BIC:", gmm_refit.bic(X_refit))

# Soft assignments and hard labels if needed
resp_refit = gmm_refit.predict_proba(X_refit)
labels_refit = gmm_refit.predict(X_refit)
print("Label counts:", np.bincount(labels_refit))

# Assuming gmm_refit is your trained model and X_refit your dataset
nll_value = negative_log_likelihood(gmm_refit, X_refit)
print("Negative Log-Likelihood:", nll_value)
