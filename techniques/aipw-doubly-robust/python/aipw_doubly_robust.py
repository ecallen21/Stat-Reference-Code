"""Augmented IPW / doubly robust ATE (Reference Sec 15.8).

Robins-Rotnitzky-Zhao 1994.  Combine two nuisance estimates:
  * PS model e(X) = P(T=1|X)
  * OUTCOME model m_t(X) = E[Y|T=t, X]

  ATE_AIPW = mean( m_1(X) - m_0(X)
                 + T(Y - m_1(X))/e(X)
                 - (1-T)(Y - m_0(X))/(1-e(X)) )

DOUBLE ROBUSTNESS: unbiased if EITHER model is correct.  Efficient
under both.  Compare with plain IPTW / g-formula.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression, LinearRegression


def aipw(X, T, y, cross_fit=False, seed=0):
    if cross_fit:
        # 2-fold cross-fitting
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        A, B = np.array_split(idx, 2)
        mA1 = LinearRegression().fit(X[A][T[A] == 1], y[A][T[A] == 1])
        mA0 = LinearRegression().fit(X[A][T[A] == 0], y[A][T[A] == 0])
        eA = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(X[A], T[A])
        mB1 = LinearRegression().fit(X[B][T[B] == 1], y[B][T[B] == 1])
        mB0 = LinearRegression().fit(X[B][T[B] == 0], y[B][T[B] == 0])
        eB = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(X[B], T[B])
        # Predict on the OTHER fold
        m1 = np.zeros(len(y)); m0 = np.zeros(len(y)); ps = np.zeros(len(y))
        m1[A] = mB1.predict(X[A]); m0[A] = mB0.predict(X[A])
        ps[A] = eB.predict_proba(X[A])[:, 1]
        m1[B] = mA1.predict(X[B]); m0[B] = mA0.predict(X[B])
        ps[B] = eA.predict_proba(X[B])[:, 1]
    else:
        m1 = LinearRegression().fit(X[T == 1], y[T == 1]).predict(X)
        m0 = LinearRegression().fit(X[T == 0], y[T == 0]).predict(X)
        ps = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(X, T).predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.02, 0.98)
    ate = float(np.mean(m1 - m0
                        + T * (y - m1) / ps
                        - (1 - T) * (y - m0) / (1 - ps)))
    return {"ATE_AIPW": ate,
            "ATE_gformula": float((m1 - m0).mean()),
            "ATE_IPTW": float((T * y / ps).mean() - ((1 - T) * y / (1 - ps)).mean())}


if __name__ == "__main__":
    print("=== AIPW / doubly robust ATE ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(0, 1, (n, 3))
    logit_T = 0.6 * X[:, 0] + 0.4 * X[:, 1]
    T = (rng.random(n) < 1 / (1 + np.exp(-logit_T))).astype(int)
    # True ATE = 0.4; nonlinear confounding
    y = 1.0 + 0.4 * T + 0.5 * X[:, 0] + 0.3 * X[:, 1] ** 2 + rng.normal(0, 1, n)

    for cf in (False, True):
        r = aipw(X, T, y, cross_fit=cf)
        print(f"  cross-fit = {cf}")
        print(f"    G-formula estimate: {r['ATE_gformula']:+.3f}")
        print(f"    IPTW      estimate: {r['ATE_IPTW']:+.3f}")
        print(f"    AIPW  DR  estimate: {r['ATE_AIPW']:+.3f}   (true ATE = 0.4)")
        print()

    print("--- library cross-check (R AIPW/CausalGAM; Python DoubleML/EconML DRLearner) ---")
