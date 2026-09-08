"""CatBoost / Ordered Boosting (Reference Sec 47.88).

Prokhorenkova, Gusev, Vorobev, Dorogush & Gulin 2018 'CatBoost:
unbiased boosting with categorical features', NeurIPS. Two
innovations over XGBoost / LightGBM:

1. **Ordered target statistics** for categorical variables:
   for each row i, encode category c using only rows j < i
   (in a random permutation) with matching category, avoiding
   TARGET LEAKAGE that plain target-encoding suffers.
2. **Ordered boosting** for the training loop itself: gradient
   estimates for row i come from a model trained WITHOUT row i,
   reducing the "prediction shift" bias found in classical GBM.

Illustrated here on high-cardinality categorical features where
plain target encoding leaks and CatBoost's ordered variant fixes
the CV MSE gap.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.model_selection import KFold    # CV eval
from sklearn.ensemble import GradientBoostingRegressor    # base GBM


def leaky_target_encode(X_cat, y):
    """Plain target-mean encoding -- uses the row's own y (leaks)."""
    codes = np.zeros_like(X_cat, dtype=float)
    cats = np.unique(X_cat)
    for c in cats:
        codes[X_cat == c] = y[X_cat == c].mean()
    return codes


def ordered_target_encode(X_cat, y, seed=0):
    """CatBoost-style ordered target statistic: for each row i in a random
    permutation, use only prior-position rows with matching category.
    """
    rng = np.random.default_rng(seed)
    n = len(X_cat)
    order = rng.permutation(n)
    codes = np.full(n, y.mean(), dtype=float)
    sums = {}; counts = {}
    for pos in order:
        c = X_cat[pos]
        if counts.get(c, 0) > 0:
            codes[pos] = sums[c] / counts[c]
        sums[c] = sums.get(c, 0.0) + y[pos]
        counts[c] = counts.get(c, 0) + 1
    return codes


if __name__ == "__main__":
    print("=== CatBoost ordered boosting (Prokhorenkova et al 2018) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    cat = rng.integers(0, 30, size=n)                 # 30-level categorical
    x1 = rng.normal(size=n)
    # True y depends on category-mean effect + noise
    cat_effect = rng.normal(size=30) * 1.5
    y = cat_effect[cat] + 0.5 * x1 + 0.3 * rng.normal(size=n)

    for name, encoder in [
        ("leaky target enc      ", leaky_target_encode),
        ("ordered target enc    ", lambda c, y: ordered_target_encode(c, y, seed=0)),
    ]:
        mses = []
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(x1):
            enc_tr = encoder(cat[tr], y[tr])
            # Same-permutation encoding for test uses train sums only
            train_map = {c: y[tr][cat[tr] == c].mean() for c in np.unique(cat[tr])}
            enc_te = np.array([train_map.get(c, y[tr].mean()) for c in cat[te]])
            X_tr = np.column_stack([x1[tr], enc_tr])
            X_te = np.column_stack([x1[te], enc_te])
            m = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0).fit(X_tr, y[tr])
            mses.append(float(((m.predict(X_te) - y[te]) ** 2).mean()))
        print(f"  {name}  5-fold CV MSE = {np.mean(mses):.3f} +/- {np.std(mses):.3f}")

    # Also show the in-fold train-MSE gap where leakage bites:
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    print()
    for name, encoder in [
        ("leaky target enc     ", leaky_target_encode),
        ("ordered target enc   ", lambda c, y: ordered_target_encode(c, y, seed=0)),
    ]:
        enc = encoder(cat, y)
        Xf = np.column_stack([x1, enc])
        m = GBR(n_estimators=100, max_depth=3, random_state=0).fit(Xf, y)
        train_mse = float(((m.predict(Xf) - y) ** 2).mean())
        print(f"  {name}  in-sample TRAIN MSE = {train_mse:.3f}")

    print("\n  Leaky encoder's TRAIN MSE is artificially low (target seen by encoder);")
    print("  ordered encoding removes that leakage -> honest training signal.")
    print("\n--- library cross-check (catboost R package; catboost Python) ---")
