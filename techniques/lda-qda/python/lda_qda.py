"""Linear and Quadratic Discriminant Analysis (Reference §9.30).

Both fit a class-conditional multivariate Gaussian to each class and classify
via Bayes' rule using the prior class probabilities pi_k.

    P(class = k | x)  proportional to  pi_k * N(x | mu_k, Sigma_k)

Difference:
    LDA : assume all Sigma_k = Sigma  (pooled covariance)   -> LINEAR decision boundary
    QDA : each class has its own Sigma_k                    -> QUADRATIC decision boundary

Decision rule (equivalently, argmax over k of):

    LDA: delta_k(x) = x' Sigma^{-1} mu_k  -  0.5 mu_k' Sigma^{-1} mu_k  +  log pi_k

    QDA: delta_k(x) = -0.5 log|Sigma_k|  -  0.5 (x - mu_k)' Sigma_k^{-1} (x - mu_k)  +  log pi_k

LDA is simpler and pays off when covariances really are similar; QDA is more
flexible but needs more data per class to estimate Sigma_k well.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def fit_lda(X, y) -> dict:
    """Fit LDA (pooled covariance)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n, p = X.shape
    K = len(classes)
    priors = counts / n
    means = np.array([X[y == c].mean(axis=0) for c in classes])
    # Pooled Sigma
    Sigma = np.zeros((p, p))
    for i, c in enumerate(classes):
        Xc = X[y == c]; diff = Xc - means[i]
        Sigma += diff.T @ diff
    Sigma /= (n - K)
    return {"classes": classes.tolist(), "priors": priors.tolist(),
            "means": means.tolist(), "Sigma_pooled": Sigma.tolist(),
            "n": n, "p": p, "K": K, "method": "LDA (pooled Sigma)"}


def predict_lda(fit, X_new) -> dict:
    """Return predicted class + posterior probabilities under LDA."""
    X_new = np.asarray(X_new, dtype=float)
    classes = np.array(fit["classes"])
    priors = np.array(fit["priors"])
    means = np.array(fit["means"])
    Sigma = np.array(fit["Sigma_pooled"])
    Sigma_inv = np.linalg.inv(Sigma)
    # delta_k(x) for each class
    deltas = np.zeros((X_new.shape[0], len(classes)))
    for k in range(len(classes)):
        m = means[k]
        deltas[:, k] = X_new @ Sigma_inv @ m - 0.5 * m @ Sigma_inv @ m + math.log(priors[k])
    # softmax to get posteriors
    mx = deltas.max(axis=1, keepdims=True)
    ex = np.exp(deltas - mx)
    post = ex / ex.sum(axis=1, keepdims=True)
    preds = classes[np.argmax(deltas, axis=1)]
    return {"predictions": preds.tolist(), "posteriors": post.tolist()}


def fit_qda(X, y) -> dict:
    """Fit QDA (per-class covariance)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n, p = X.shape
    K = len(classes)
    priors = counts / n
    means = np.array([X[y == c].mean(axis=0) for c in classes])
    Sigmas = []
    for i, c in enumerate(classes):
        Xc = X[y == c]
        diff = Xc - means[i]
        Sigmas.append((diff.T @ diff) / (Xc.shape[0] - 1))
    return {"classes": classes.tolist(), "priors": priors.tolist(),
            "means": means.tolist(),
            "Sigmas": [S.tolist() for S in Sigmas],
            "n": n, "p": p, "K": K, "method": "QDA (per-class Sigma)"}


def predict_qda(fit, X_new) -> dict:
    """Predicted class + posteriors under QDA."""
    X_new = np.asarray(X_new, dtype=float)
    classes = np.array(fit["classes"])
    priors = np.array(fit["priors"])
    means = np.array(fit["means"])
    Sigmas = [np.array(S) for S in fit["Sigmas"]]
    deltas = np.zeros((X_new.shape[0], len(classes)))
    for k in range(len(classes)):
        S = Sigmas[k]
        sign, logdet = np.linalg.slogdet(S)
        diff = X_new - means[k]
        S_inv = np.linalg.inv(S)
        quad = np.einsum("ij,jk,ik->i", diff, S_inv, diff)
        deltas[:, k] = -0.5 * logdet - 0.5 * quad + math.log(priors[k])
    mx = deltas.max(axis=1, keepdims=True)
    ex = np.exp(deltas - mx)
    post = ex / ex.sum(axis=1, keepdims=True)
    preds = classes[np.argmax(deltas, axis=1)]
    return {"predictions": preds.tolist(), "posteriors": post.tolist()}


def library_versions(X, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis().fit(X, y)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    return {"sklearn LDA priors": lda.priors_.tolist(),
            "sklearn LDA means": lda.means_.tolist(),
            "sklearn LDA train accuracy": float(lda.score(X, y)),
            "sklearn QDA train accuracy": float(qda.score(X, y))}


if __name__ == "__main__":
    rng = np.random.default_rng(89)
    # 3-class problem in 2D with overlapping Gaussians
    n_per = 80
    mu_true = np.array([[0, 0], [3, 3], [6, 0]])
    Sigma_shared = np.array([[1.0, 0.3], [0.3, 1.0]])
    X = np.vstack([rng.multivariate_normal(m, Sigma_shared, n_per) for m in mu_true])
    y = np.repeat([0, 1, 2], n_per)

    lda = fit_lda(X, y)
    qda = fit_qda(X, y)
    lda_pred = predict_lda(lda, X)
    qda_pred = predict_qda(qda, X)

    acc_l = float((np.array(lda_pred["predictions"]) == y).mean())
    acc_q = float((np.array(qda_pred["predictions"]) == y).mean())
    print(f"=== LDA (pooled Sigma), train accuracy: {acc_l:.4f} ===")
    print(f"  priors: {lda['priors']}")
    print(f"  means: {lda['means']}")

    print(f"\n=== QDA, train accuracy: {acc_q:.4f} ===")
    print(f"  priors: {qda['priors']}")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(X, y).items():
        print(f"  {k}: {v}")
