"""Naive Bayes classifier (Reference §26.10).

Bayes rule + CONDITIONAL INDEPENDENCE assumption:
    Pr(y = c | x) proportional to  Pr(y = c) * prod_j Pr(x_j | y = c)

The "naive" bit is treating features as independent given the class.  In
practice this is wildly wrong but often works well for text and other
sparse categorical data.

Gaussian NB (continuous features)
    Pr(x_j | y = c) = N(mu_jc, sigma_jc^2)
    Estimate mu_jc, sigma_jc from within-class means/variances.

Multinomial NB (bag-of-words counts)
    Pr(w_j | y = c) = (count_jc + alpha) / (sum_j count_jc + alpha V)
    Laplace smoothing alpha > 0 prevents zero-probabilities.

Prediction: pick c maximizing log Pr(y = c) + sum_j log Pr(x_j | y = c).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def gaussian_nb(X, y) -> dict:
    """Gaussian NB with per-class per-feature mean and variance."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    classes = np.unique(y)
    priors = {c: float((y == c).mean()) for c in classes}
    stats_per_class = {}
    for c in classes:
        Xc = X[y == c]
        stats_per_class[c] = {"mean": Xc.mean(0), "var": Xc.var(0) + 1e-6}
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        out = []
        for x in X_new:
            best_c, best_lp = None, -np.inf
            for c in classes:
                mu = stats_per_class[c]["mean"]; v = stats_per_class[c]["var"]
                lp = math.log(priors[c])
                lp += -0.5 * np.sum(np.log(2 * math.pi * v) + (x - mu) ** 2 / v)
                if lp > best_lp: best_lp = lp; best_c = c
            out.append(best_c)
        return np.array(out)
    return {"priors": priors, "stats_per_class": stats_per_class,
            "predict": predict, "classes": classes,
            "method": "Gaussian Naive Bayes"}


def multinomial_nb(X, y, alpha: float = 1.0) -> dict:
    """Multinomial NB for count features with Laplace smoothing."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    classes = np.unique(y); n, V = X.shape
    log_prior = {}; log_theta = {}
    for c in classes:
        Xc = X[y == c]
        log_prior[c] = math.log((y == c).mean())
        col_sums = Xc.sum(0) + alpha
        log_theta[c] = np.log(col_sums / col_sums.sum())
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        out = []
        for x in X_new:
            best_c, best_lp = None, -np.inf
            for c in classes:
                lp = log_prior[c] + float(x @ log_theta[c])
                if lp > best_lp: best_lp = lp; best_c = c
            out.append(best_c)
        return np.array(out)
    return {"log_prior": log_prior, "log_theta": log_theta,
            "predict": predict, "classes": classes,
            "method": "Multinomial Naive Bayes"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Gaussian NB on 3 Gaussian blobs ===")
    X = np.vstack([rng.normal([0, 0], 1, (100, 2)),
                    rng.normal([4, 0], 1, (100, 2)),
                    rng.normal([2, 4], 1, (100, 2))])
    y = np.repeat([0, 1, 2], 100)
    fit = gaussian_nb(X, y)
    print(f"  training accuracy = {(fit['predict'](X) == y).mean():.3f}")

    print("\n=== Multinomial NB on synthetic bag-of-words ===")
    V = 30; n_per = 50; classes = 3
    Xc = []
    for c in range(classes):
        theta_c = rng.dirichlet(np.ones(V) * 0.5)
        Xc.append(rng.multinomial(20, theta_c, size=n_per))
    X = np.vstack(Xc); y = np.repeat(range(classes), n_per)
    fit = multinomial_nb(X, y, alpha=1.0)
    print(f"  training accuracy = {(fit['predict'](X) == y).mean():.3f}")

    print("\n--- library cross-check (sklearn GaussianNB) ---")
    try:
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        # Gaussian
        X = np.vstack([rng.normal([0, 0], 1, (100, 2)), rng.normal([4, 0], 1, (100, 2)),
                        rng.normal([2, 4], 1, (100, 2))])
        y = np.repeat([0, 1, 2], 100)
        print(f"  sklearn GaussianNB accuracy = {GaussianNB().fit(X, y).score(X, y):.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
