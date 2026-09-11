"""AutoGluon - Automated Machine Learning (Reference Sec 47.243).

Erickson et al 2020 'AutoGluon-Tabular: Robust and Accurate AutoML
for Structured Data'. Zero-hyperparameter AutoML for tabular data:

    1. Train MANY diverse models (GBM variants, RF, NN, linear).
    2. STACK them: level-1 models feed predictions into a level-2
       meta-learner (weighted average or another GBM).
    3. Bagging + k-fold ensembling.

Consistently top-ranks on Kaggle / OpenML with a single fit()
call. Sibling technique: FLAML (Wang 2021, cost-frugal automl).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def train_base_models(X, y, seed=0):
    """Train a small stack of diverse models."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=50, random_state=seed),
        "gbm": GradientBoostingRegressor(n_estimators=50, random_state=seed),
        "knn": KNeighborsRegressor(n_neighbors=5),
    }
    for m in models.values(): m.fit(X, y)
    return models


def stack_predict(models, X, weights=None):
    """Weighted average of level-1 predictions."""
    preds = np.array([m.predict(X) for m in models.values()])
    if weights is None: weights = np.ones(len(models)) / len(models)
    return weights @ preds


def fit_stack_weights(models, X_val, y_val):
    """Fit level-2 weights via non-negative LS on validation predictions."""
    preds = np.array([m.predict(X_val) for m in models.values()]).T   # (n_val, K)
    # Ridge-regularised NNLS-like: solve preds @ w = y_val with w >= 0
    from scipy.optimize import nnls
    w, _ = nnls(preds, y_val)
    if w.sum() > 0: w = w / w.sum()
    else: w = np.ones(len(models)) / len(models)
    return w


if __name__ == "__main__":
    print("=== AutoGluon-style AutoML (Erickson et al 2020) ===\n")
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.25, random_state=0)

    models = train_base_models(Xtr2, ytr2)
    print("  Base-level models: " + ", ".join(models.keys()))
    for name, m in models.items():
        mse = float(np.mean((m.predict(Xte) - yte) ** 2))
        print(f"    {name:>5s} test MSE = {mse:.1f}")

    # Uniform stack
    p_uniform = stack_predict(models, Xte)
    mse_uniform = float(np.mean((p_uniform - yte) ** 2))
    # Fitted stack
    w = fit_stack_weights(models, Xval, yval)
    p_stacked = stack_predict(models, Xte, weights=w)
    mse_stacked = float(np.mean((p_stacked - yte) ** 2))
    print(f"\n  Stack weights (fitted on val):")
    for k, name in enumerate(models.keys()):
        print(f"    {name:>5s} weight = {w[k]:.3f}")
    print(f"\n  Uniform-weighted stack MSE:   {mse_uniform:.1f}")
    print(f"  Fitted-weighted stack MSE:    {mse_stacked:.1f}")

    print("\n--- library cross-check (autogluon-tabular; flaml.AutoML; h2o.automl) ---")
