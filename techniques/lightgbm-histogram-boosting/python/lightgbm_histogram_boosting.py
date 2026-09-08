"""LightGBM histogram + leaf-wise boosting (Reference Sec 47.105).

Ke, Meng, Finley, Wang, Chen, Ma, Ye & Liu 2017 'LightGBM: A highly
efficient gradient boosting decision tree', NeurIPS. Contrast with
XGBoost:

    * HISTOGRAM-based split finding: bin features to 256 values;
      O(#data * #features) preprocessing then O(#bins * #features)
      per split.
    * LEAF-WISE (best-first) tree growth vs level-wise; deeper
      trees at same #leaves.
    * GOSS: sample large-|gradient| rows first.
    * EFB: bundle exclusive sparse features.

Demonstrated here via `sklearn.ensemble.HistGradientBoostingRegressor`
which shares the histogram-based split-finding idea (LightGBM
reference not installed here).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
import time    # wall-clock
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    print("=== LightGBM / Histogram GBM (Ke et al 2017) ===\n")
    rng = np.random.default_rng(0)
    n, p = 20_000, 20
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + np.sin(2 * X[:, 3]) + 0.5 * rng.normal(size=n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    # Baseline: exact-split GBM
    t0 = time.time()
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                       learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    exact_time = time.time() - t0
    exact_mse = float(((yte - gbm.predict(Xte)) ** 2).mean())

    # Histogram GBM (LightGBM-style)
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=8,
                                          learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    hist_time = time.time() - t0
    hist_mse = float(((yte - hgb.predict(Xte)) ** 2).mean())

    print(f"  Exact-split GBM (n_tr={len(Xtr)}, T=300):")
    print(f"    fit time = {exact_time:5.2f} s   test MSE = {exact_mse:.3f}")
    print(f"  Histogram GBM (leaf-wise):")
    print(f"    fit time = {hist_time:5.2f} s   test MSE = {hist_mse:.3f}   speedup = {exact_time/hist_time:.1f}x")

    try:
        import lightgbm as lgb
        t0 = time.time()
        lg = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.1,
                                 verbosity=-1, random_state=0).fit(Xtr, ytr)
        lg_time = time.time() - t0
        lg_mse = float(((yte - lg.predict(Xte)) ** 2).mean())
        print(f"  Native LightGBM:")
        print(f"    fit time = {lg_time:5.2f} s   test MSE = {lg_mse:.3f}")
    except ImportError:
        print("\n  lightgbm not installed. Install:  pip install lightgbm")

    print("\n--- library cross-check (lightgbm R + Python; sklearn HistGradientBoosting) ---")
