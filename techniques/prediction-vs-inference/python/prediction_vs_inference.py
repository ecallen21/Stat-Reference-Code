"""Prediction vs inference (Reference Sec 39.1).

Two DIFFERENT goals with DIFFERENT model choices (Shmueli 2010).

  INFERENCE:  estimate a coefficient / causal effect.  Include
              confounders regardless of their marginal explanatory
              power; report SE and CI; use the interpretable model
              class.

  PREDICTION: minimise out-of-sample error for a NEW observation.
              Include predictors that improve out-of-sample performance
              even if individual coefficients are noisy; regularise
              to control overfitting; interpretability is optional.

Demonstrated here:

  * Confounder-inclusion decision by CHANGE-IN-ESTIMATE (inference)
    vs by CROSS-VALIDATED MSE (prediction).

  * A high-VIF, mildly-informative predictor may hurt inference
    (huge SE) but help prediction (lower CV error).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.linear_model import LinearRegression, RidgeCV    # simple models
from sklearn.model_selection import cross_val_score           # CV MSE


def cie_confounder_screen(x, z, y, threshold=0.10):
    """Change-in-estimate: keep z if inclusion changes beta_x by > threshold * beta_x."""
    beta_wo = LinearRegression().fit(x[:, None], y).coef_[0]
    beta_w = LinearRegression().fit(np.column_stack([x, z]), y).coef_[0]
    change = abs(beta_w - beta_wo) / max(abs(beta_wo), 1e-9)
    return {"beta_x_alone": float(beta_wo), "beta_x_adj": float(beta_w),
            "rel_change": float(change), "keep_z_for_inference": change > threshold}


if __name__ == "__main__":
    print("=== Prediction vs inference: same model class, different decisions ===\n")
    rng = np.random.default_rng(0)
    n = 400
    # z is a confounder: strongly related to both x and y
    z = rng.normal(0, 1, n)
    x = 0.6 * z + rng.normal(0, 0.5, n)             # x correlated with z
    y = 1.0 * x + 2.0 * z + rng.normal(0, 1.0, n)    # true beta_x = 1

    # INFERENCE view: change-in-estimate rule
    cie = cie_confounder_screen(x, z, y, threshold=0.10)
    print(f"  INFERENCE (change-in-estimate for beta_x)")
    print(f"    beta_x alone     = {cie['beta_x_alone']:.3f}"
          f"   beta_x adj for z = {cie['beta_x_adj']:.3f}")
    print(f"    Relative change  = {cie['rel_change']:.3f}"
          f"   -> {'INCLUDE z' if cie['keep_z_for_inference'] else 'drop z'}")

    # PREDICTION view: CV MSE
    X_solo = x[:, None]
    X_with = np.column_stack([x, z])
    mse_solo = -cross_val_score(LinearRegression(), X_solo, y,
                                cv=10, scoring="neg_mean_squared_error").mean()
    mse_with = -cross_val_score(LinearRegression(), X_with, y,
                                cv=10, scoring="neg_mean_squared_error").mean()
    print(f"\n  PREDICTION (10-fold CV MSE)")
    print(f"    x only           = {mse_solo:.3f}")
    print(f"    x + z            = {mse_with:.3f}"
          f"   -> {'INCLUDE z' if mse_with < mse_solo else 'drop z'}")

    # Case where prediction and inference disagree: noisy but complementary predictor u
    u = rng.normal(0, 1, n)                          # unrelated to y
    X_with_u = np.column_stack([x, u])
    mse_with_u = -cross_val_score(LinearRegression(), X_with_u, y,
                                  cv=10, scoring="neg_mean_squared_error").mean()
    cie_u = cie_confounder_screen(x, u, y, threshold=0.10)
    print(f"\n  Contrast: noise predictor u (unrelated to y, correlated with x)")
    print(f"    Inference : beta_x change {cie_u['rel_change']:.3f}"
          f"   -> {'INCLUDE u (spurious!)' if cie_u['keep_z_for_inference'] else 'drop u'}")
    print(f"    Prediction: CV MSE {mse_with_u:.3f} vs {mse_solo:.3f}"
          f"   -> {'INCLUDE u' if mse_with_u < mse_solo else 'drop u (correct)'}")

    print("\n  Conclusion:")
    print("    * Include confounders for INFERENCE regardless of predictive power.")
    print("    * Include predictors for PREDICTION only if they improve CV error.\n")
    print("--- library cross-check (R rms/caret; Python sklearn/statsmodels) ---")
