"""XGBoost (Reference Sec 47.104).

Chen & Guestrin 2016 'XGBoost: a scalable tree boosting system',
KDD. Gradient boosting with:

    * SECOND-ORDER (Newton) TAYLOR expansion of the loss
    * REGULARISED objective: 0.5 lambda ||w||^2 + gamma T  (T = # leaves)
    * Sparse-aware split finding + histogram / approximate splits
    * Column-block cache + parallelisation.

Second-order updates give the closed-form optimal leaf weight
w* = -G / (H + lambda) and split gain
    Gain = 0.5 [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G_S^2/(H_S+lambda)] - gamma.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

try:
    import xgboost as xgb    # library reference
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


def newton_boost_squared(X, y, n_trees=100, max_depth=3, lr=0.1, lam=1.0, gam=0.0, seed=0):
    """Toy XGBoost-style second-order boosting for squared loss.
    Regression tree with regularised leaf weights w* = -G / (H + lam).
    """
    from sklearn.tree import DecisionTreeRegressor    # weak learner
    rng = np.random.default_rng(seed)
    n = len(y)
    f = np.zeros(n)
    trees = []
    for t in range(n_trees):
        g = f - y                              # d L / d f  for L = 0.5 (y - f)^2
        h = np.ones(n)                          # d^2 L / d f^2
        # Fit tree that predicts w* = -g / (h + lam)  via targeting -g / (h+lam) with LS
        target = -g / (h + lam)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=t).fit(X, target)
        f = f + lr * tree.predict(X)
        trees.append(tree)
    return {"trees": trees, "lr": lr, "yhat": f}


def predict_boost(trees, lr, X):
    return sum(lr * t.predict(X) for t in trees)


if __name__ == "__main__":
    print("=== XGBoost / Newton boosting (Chen-Guestrin 2016) ===\n")
    rng = np.random.default_rng(0)
    n, p = 800, 8
    X = rng.normal(size=(n, p))
    y = 2 * X[:, 0] - X[:, 1] * X[:, 2] + np.sin(2 * X[:, 3]) + 0.3 * rng.normal(size=n)
    train, test = slice(0, 600), slice(600, n)

    for T in [50, 200, 500]:
        m = newton_boost_squared(X[train], y[train], n_trees=T, max_depth=3, lr=0.1, lam=1.0)
        yhat = predict_boost(m["trees"], m["lr"], X[test])
        mse = float(((y[test] - yhat) ** 2).mean())
        print(f"  From-scratch Newton boost T={T:3d}  test MSE = {mse:.3f}")

    if HAVE_XGB:
        for T in [50, 200, 500]:
            m = xgb.XGBRegressor(n_estimators=T, max_depth=3, learning_rate=0.1,
                                    reg_lambda=1.0, verbosity=0).fit(X[train], y[train])
            mse = float(((y[test] - m.predict(X[test])) ** 2).mean())
            print(f"  xgboost.XGBRegressor T={T:3d}   test MSE = {mse:.3f}")
    else:
        print("\n  xgboost not installed. Install:  pip install xgboost")

    print("\n--- library cross-check (xgboost R; xgboost / lightgbm / catboost Python) ---")
