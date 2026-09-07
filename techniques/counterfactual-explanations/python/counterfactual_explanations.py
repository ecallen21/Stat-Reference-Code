"""Counterfactual explanations (Reference Sec 47.33).

Wachter, Mittelstadt & Russell 2017 'Counterfactual explanations
without opening the black box: automated decisions and the GDPR'.
Given a point x with prediction f(x) = y_bad, find the SMALLEST
change x' -> y_desired:

    x' = argmin  d(x, x')  +  lambda * L(f(x'), y_desired)

Common distances: L1 (encourages sparse changes), MAD-normalised
L1 (Wachter), Gower for mixed types.

Actionable, individual-level explanation (used in GDPR-style
right-to-explanation contexts).

We implement a simple gradient / grid search counterfactual for a
differentiable logistic classifier.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def logistic_predict(x, beta):
    return 1 / (1 + np.exp(-x @ beta))


def find_counterfactual(x_star, beta, y_desired=1.0, threshold=0.5,
                         lam=1.0, lr=0.05, epochs=300, l1_scale=None):
    """Wachter-style minimisation of d(x, x') + lam * (f(x') - y_desired)^2."""
    x = x_star.copy()
    if l1_scale is None:
        l1_scale = np.ones_like(x_star)      # unit MAD
    for _ in range(epochs):
        p = logistic_predict(x, beta)
        loss_pred = 0.5 * (p - y_desired) ** 2
        grad_pred = (p - y_desired) * p * (1 - p) * beta
        #  L1 distance -> subgradient sign((x - x*) / MAD)
        grad_dist = np.sign(x - x_star) / l1_scale
        x = x - lr * (grad_dist + lam * grad_pred)
    return {"x_cf": x, "delta": x - x_star,
            "prob_after": float(logistic_predict(x, beta)),
            "class_after": int(logistic_predict(x, beta) > threshold)}


if __name__ == "__main__":
    print("=== Counterfactual explanations (Wachter 2017) ===\n")
    #  Logistic classifier: applicant approved if beta * x > 0
    beta = np.array([1.2, -0.6, 0.3, 0.8])   # feature weights
    feature_names = ["income", "debt", "credit_history_years", "employment_years"]
    x_star = np.array([-1.0, 1.5, 0.5, 0.2])   # applicant currently rejected
    p_star = float(logistic_predict(x_star, beta))
    print(f"  Applicant x* = {x_star.tolist()}")
    print(f"  Predicted probability = {p_star:.3f} (rejected)\n")

    r = find_counterfactual(x_star, beta, y_desired=1.0, lam=20.0, lr=0.03, epochs=600)
    print(f"  Counterfactual x_cf = {r['x_cf'].round(3).tolist()}")
    print(f"  Post prob = {r['prob_after']:.3f}   Approved: {bool(r['class_after'])}")
    print(f"\n  Feature changes:")
    for name, dx in zip(feature_names, r['delta']):
        print(f"    Delta {name:22s} = {dx:+.3f}")

    print("\n--- library cross-check (alibi.explainers.CounterfactualProto Python; iml / lime R) ---")
