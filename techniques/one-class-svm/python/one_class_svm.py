"""One-class SVM for novelty / anomaly detection (Reference §21.x extra).

Schölkopf et al. (2001): find a hyperplane that separates the training data
from the origin in the RBF-kernel feature space with maximum margin.

Primal:
    min_{w, rho, xi}  1/2 ||w||^2 + 1/(nu n) sum xi_i - rho
    s.t.  <w, phi(x_i)> >= rho - xi_i,   xi_i >= 0

Decision:
    f(x) = sign(<w, phi(x)> - rho)
        (+1 = inlier, -1 = outlier)

nu is an upper bound on the fraction of outliers and a lower bound on the
fraction of support vectors.

Here we defer to scikit-learn (cvxopt is heavier than the scope of this repo);
provide a from-scratch KDE-baseline anomaly detector as a companion.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (log, pi)

import numpy as np    # numerical arrays + linear algebra


def kde_anomaly_score(X_train, X_test, bandwidth: float = None) -> np.ndarray:
    """Negative log-density of the RBF-Gaussian KDE fit on X_train."""
    X_train = np.asarray(X_train, dtype=float); X_test = np.asarray(X_test, dtype=float)
    n, d = X_train.shape
    if bandwidth is None:
        sd = X_train.std(axis=0, ddof=1).mean()
        bandwidth = sd * n ** (-1.0 / (d + 4))            # Scott's rule
    # log-density of test points under Gaussian KDE (log-sum-exp)
    log_scores = []
    const = -0.5 * d * math.log(2 * math.pi * bandwidth ** 2)
    for xt in X_test:
        d2 = ((X_train - xt) ** 2).sum(axis=1)
        lp = -0.5 * d2 / bandwidth ** 2 + const
        m = lp.max(); log_scores.append(m + math.log(np.exp(lp - m).mean()))
    return -np.asarray(log_scores)


def kde_threshold(scores_train, contamination: float = 0.05) -> float:
    return float(np.quantile(scores_train, 1 - contamination))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    X_train = rng.normal(size=(n, 2))                     # inliers ~ N(0, I)
    X_test = np.vstack([rng.normal(size=(50, 2)),
                        rng.uniform(-6, 6, (50, 2))])     # 50 in-dist + 50 anomalies
    y_test = np.hstack([np.ones(50), -np.ones(50)])       # +1 in, -1 out

    print("=== KDE-baseline anomaly detector (Scott bandwidth) ===")
    s_train = kde_anomaly_score(X_train, X_train)
    s_test = kde_anomaly_score(X_train, X_test)
    thr = kde_threshold(s_train, contamination=0.05)
    y_hat = np.where(s_test > thr, -1, 1)
    tpr = float(((y_hat == -1) & (y_test == -1)).sum() / (y_test == -1).sum())
    fpr = float(((y_hat == -1) & (y_test ==  1)).sum() / (y_test ==  1).sum())
    print(f"  threshold (5% train tail) = {thr:.3f}")
    print(f"  outlier TPR = {tpr:.3f}   inlier FPR = {fpr:.3f}")

    print("\n=== One-class SVM (sklearn RBF, nu=0.05) ===")
    try:
        from sklearn.svm import OneClassSVM
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train)
        y_hat = oc.predict(X_test)
        tpr = float(((y_hat == -1) & (y_test == -1)).sum() / (y_test == -1).sum())
        fpr = float(((y_hat == -1) & (y_test ==  1)).sum() / (y_test ==  1).sum())
        print(f"  outlier TPR = {tpr:.3f}   inlier FPR = {fpr:.3f}")
        # nu should approximately equal the fraction of support vectors
        sv_frac = oc.support_.shape[0] / n
        print(f"  fraction of support vectors = {sv_frac:.3f}   (~nu = 0.05)")
    except ImportError:
        print("  (sklearn not installed; KDE baseline only)")
