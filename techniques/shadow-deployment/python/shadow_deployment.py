"""Shadow deployment (Reference Ch 32 MLOps).

Route production traffic through BOTH the CURRENT (production) model and
a CANDIDATE (shadow) model. Only the CURRENT model's predictions are
returned to the user; the candidate's predictions are LOGGED for offline
comparison.

Advantages:
  * Zero user-facing risk.
  * Real production traffic (not just held-out data).
  * Direct measurement of prediction disagreement + candidate-vs-current
    accuracy (once labels arrive).

Standard summaries:
  * Prediction AGREEMENT rate.
  * DISAGREEMENT rate segmented by feature slice.
  * Per-model accuracy / calibration on labeled subset.

Here we implement a tiny ShadowRouter that scores incoming rows with
both models, logs a compact record, and reports the standard shadow
comparison table on a synthetic stream where the candidate is genuinely
better on part of the input space.
"""
from __future__ import annotations    # stdlib

from dataclasses import dataclass, field    # data class for records
from typing import Callable, List           # type hints

import numpy as np    # numerical arrays


@dataclass
class ShadowRecord:
    t: int
    x: np.ndarray
    y_true: int | None
    prod_pred: int
    shadow_pred: int
    prod_prob: float
    shadow_prob: float


class ShadowRouter:
    def __init__(self, prod_fn: Callable, shadow_fn: Callable):
        self.prod_fn = prod_fn
        self.shadow_fn = shadow_fn
        self.records: List[ShadowRecord] = []

    def serve(self, x, t, y_true=None):
        prod_prob = float(self.prod_fn(x))
        shadow_prob = float(self.shadow_fn(x))
        self.records.append(ShadowRecord(t=t, x=x, y_true=y_true,
                                           prod_pred=int(prod_prob > 0.5),
                                           shadow_pred=int(shadow_prob > 0.5),
                                           prod_prob=prod_prob,
                                           shadow_prob=shadow_prob))
        # Return the PRODUCTION prediction only.
        return int(prod_prob > 0.5)

    def report(self):
        n = len(self.records)
        if n == 0:
            return {}
        agree = sum(r.prod_pred == r.shadow_pred for r in self.records) / n
        prob_diff = np.mean([abs(r.prod_prob - r.shadow_prob) for r in self.records])
        with_labels = [r for r in self.records if r.y_true is not None]
        acc_prod = float(np.mean([r.prod_pred == r.y_true for r in with_labels])) if with_labels else float("nan")
        acc_shadow = float(np.mean([r.shadow_pred == r.y_true for r in with_labels])) if with_labels else float("nan")
        # Segment analysis: disagreement by feature-0 sign.
        seg_pos = [r for r in self.records if r.x[0] > 0]
        seg_neg = [r for r in self.records if r.x[0] <= 0]
        dis_pos = float(np.mean([r.prod_pred != r.shadow_pred for r in seg_pos])) if seg_pos else 0.0
        dis_neg = float(np.mean([r.prod_pred != r.shadow_pred for r in seg_neg])) if seg_neg else 0.0
        return {"n": n, "agreement": float(agree),
                "mean_prob_diff": float(prob_diff),
                "prod_accuracy": acc_prod,
                "shadow_accuracy": acc_shadow,
                "disagree_x0_pos": dis_pos, "disagree_x0_neg": dis_neg}


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    print("=== Shadow deployment: log-only candidate comparison ===\n")
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(0, 1, (n, 3))
    beta_true = np.array([1.5, -0.5, 0.7])
    y = (_sigmoid(X @ beta_true + rng.normal(0, 0.3, n)) > 0.5).astype(int)

    # Production model: fitted on OLD data (slight coefficient mismatch).
    beta_prod = np.array([0.6, 0.0, 0.2])           # stale coefficients
    # Shadow candidate: closer to true.
    beta_shadow = np.array([1.4, -0.45, 0.7])

    router = ShadowRouter(
        prod_fn=lambda x: _sigmoid(x @ beta_prod),
        shadow_fn=lambda x: _sigmoid(x @ beta_shadow),
    )
    for t in range(n):
        router.serve(X[t], t, y_true=y[t])

    rep = router.report()
    for k, v in rep.items():
        print(f"    {k:20s}   {v}")

    print("\n  Candidate shadow accuracy > production accuracy AND high agreement"
          "\n  means it is safe to promote via canary (see canary-deployment).\n")
    print("--- library cross-check (seldon-deploy shadow deployments; kubeflow serving) ---")
