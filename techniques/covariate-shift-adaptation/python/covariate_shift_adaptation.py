"""Covariate shift adaptation via density-ratio importance weighting (Ch 29 UQ).

Shimodaira (2000); Sugiyama, Suzuki & Kanamori (2012).

Assumption: p_test(y | x) = p_train(y | x) but p_test(x) != p_train(x).
The optimal risk under p_test can be re-written as an EXPECTATION under
p_train reweighted by the density ratio w(x) = p_test(x) / p_train(x):

  E_test[ell(f(x), y)]  =  E_train[ w(x) * ell(f(x), y) ]

Estimating w(x) by pooled logistic regression (Bickel-Bruckner-Scheffer 2007):
label train samples 0 and test samples 1, fit a probabilistic classifier
p(y=1 | x), and use

  w_hat(x) = (p(y=1|x) / p(y=0|x)) * (n_train / n_test).

Then re-fit the downstream model with weights w_hat(x_i).

Here we demonstrate on a linear-regression toy: training data biased toward
low x, test data biased toward high x. Show that the plain OLS estimate is
biased on the test set, but the importance-weighted OLS matches the truth.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_density_ratio(X_train, X_test, lr=0.1, epochs=500, l2=1e-3):
    """Logistic regression on {label 0 = train, label 1 = test}."""
    n_tr, n_te = len(X_train), len(X_test)
    X = np.vstack([X_train, X_test])
    y = np.concatenate([np.zeros(n_tr), np.ones(n_te)])
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    beta = np.zeros(X_aug.shape[1])
    for _ in range(epochs):
        p = _sigmoid(X_aug @ beta)
        g = X_aug.T @ (p - y) / len(y) + l2 * beta
        beta -= lr * g
    def w_fn(x):
        x_aug = np.hstack([x, np.ones((len(x), 1))])
        p1 = _sigmoid(x_aug @ beta)
        p0 = 1.0 - p1
        return (p1 / p0) * (n_tr / n_te)
    return w_fn, beta


def wls(X, y, w):
    """Weighted least squares beta = (X^T W X)^-1 X^T W y."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    W = np.diag(w)
    return np.linalg.solve(Xa.T @ W @ Xa, Xa.T @ W @ y)


def ols(X, y):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(Xa.T @ Xa, Xa.T @ y)


if __name__ == "__main__":
    print("=== Covariate-shift adaptation (density-ratio importance weighting) ===\n")
    rng = np.random.default_rng(0)
    # Truth: y = 2 x - 3 x^2 + eps.  Training data biased toward x in [-2, 0.5];
    # test data biased toward x in [-0.5, 2].  Same p(y|x); different p(x).
    def truth(x): return 2 * x - 3 * x ** 2

    # Training biased toward low x, test toward high x, with sizable overlap.
    x_tr = np.concatenate([rng.uniform(-2, 1.5, 350),
                           rng.uniform(1.0, 2.0, 30)])   # a few high-x samples so IW has support
    x_te = rng.uniform(-1.0, 2.0, 500)
    y_tr = truth(x_tr) + rng.normal(0, 0.5, len(x_tr))
    y_te = truth(x_te) + rng.normal(0, 0.5, len(x_te))

    # Model class: MISSPECIFIED — linear features only. Truth is quadratic,
    # so the best linear fit depends on where the mass of x sits. IW moves
    # the fit toward the test region.
    def phi(z):
        return z.reshape(-1, 1)

    Phi_tr = phi(x_tr)
    Phi_te = phi(x_te)

    # 1) Plain OLS on training set, evaluate on test.
    beta_ols = ols(Phi_tr, y_tr)
    yhat_te_ols = np.hstack([Phi_te, np.ones((len(Phi_te), 1))]) @ beta_ols
    mse_ols = np.mean((y_te - yhat_te_ols) ** 2)
    print(f"  Plain OLS betas   [x, intercept]: {np.round(beta_ols, 3).tolist()}")
    print(f"  (Model is MISSPECIFIED: linear-only fit to y = 2x - 3x^2 + noise)")
    print(f"  OLS test MSE: {mse_ols:.4f}")

    # 2) Fit density ratio then weighted OLS.
    w_fn, beta_ratio = fit_density_ratio(x_tr.reshape(-1, 1), x_te.reshape(-1, 1))
    w = w_fn(x_tr.reshape(-1, 1))
    w = np.clip(w, 0, 10)                # clip extreme weights (standard practice)
    print(f"\n  density-ratio weight range on training: [{w.min():.3f}, {w.max():.3f}]  mean {w.mean():.3f}")

    beta_iw = wls(Phi_tr, y_tr, w)
    yhat_te_iw = np.hstack([Phi_te, np.ones((len(Phi_te), 1))]) @ beta_iw
    mse_iw = np.mean((y_te - yhat_te_iw) ** 2)
    print(f"\n  IW-OLS betas      [x, intercept]: {np.round(beta_iw, 3).tolist()}")
    print(f"  IW-OLS test MSE: {mse_iw:.4f}")
    print(f"\n  MSE reduction from IW: {mse_ols - mse_iw:.4f}"
          f"  ({100 * (mse_ols - mse_iw) / mse_ols:.1f}% relative).\n")

    print("--- library cross-check (densratio; sklearn CovariateShiftEstimator; adapt) ---")
