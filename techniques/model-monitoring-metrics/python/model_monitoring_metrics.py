"""Model monitoring metrics (Reference Ch 32 MLOps).

Rolling-window monitoring of a deployed model with alert triggers on:
  * ACCURACY / F1        (needs eventual labels)
  * BRIER / LOG-LOSS     (probability quality)
  * ECE                  (calibration drift)
  * LATENCY              (system health; see inference-latency-profiling)

Two alert patterns:

  ABSOLUTE  -- fire when rolling_metric CROSSES a fixed operational threshold.
  RELATIVE  -- fire when rolling_metric is BEYOND k std of its own baseline
                (a la Shewhart / EWMA control chart).

Here we implement a small MetricLogger + rolling window + EWMA baseline +
alert triggers and run it on a simulated 2000-example stream that
experiences a mid-stream accuracy drop.
"""
from __future__ import annotations    # stdlib

from collections import deque   # rolling window container

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ece(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0; n = len(y)
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            err += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return err


class Monitor:
    def __init__(self, window=200, baseline_alpha=0.05, k_sigma=3.0,
                 alert_thresholds=None):
        self.window = window
        self.k_sigma = k_sigma
        self.probs, self.y = deque(maxlen=window), deque(maxlen=window)
        self.baseline_mean = None      # EWMA of metric
        self.baseline_var = None       # EWMA of squared deviation
        self.alpha = baseline_alpha
        self.absolute = alert_thresholds or {}
        self.log = []

    def update(self, prob_row, y_row):
        self.probs.append(prob_row); self.y.append(int(y_row))
        if len(self.probs) < self.window:
            return None
        probs = np.array(self.probs); y = np.array(self.y)
        acc = float((probs.argmax(axis=1) == y).mean())
        brier = float(((probs - np.eye(probs.shape[1])[y]) ** 2).sum(axis=1).mean())
        e = float(ece(probs, y))
        row = {"acc": acc, "brier": brier, "ece": e}
        # EWMA baseline on acc (the illustrative metric).
        if self.baseline_mean is None:
            self.baseline_mean = acc
            self.baseline_var = 1e-4
        else:
            delta = acc - self.baseline_mean
            self.baseline_mean += self.alpha * delta
            self.baseline_var  += self.alpha * (delta ** 2 - self.baseline_var)
        sd = np.sqrt(self.baseline_var + 1e-12)
        row["baseline_mean"] = float(self.baseline_mean)
        row["baseline_sd"] = float(sd)
        # Alerts
        alerts = []
        if "min_acc" in self.absolute and acc < self.absolute["min_acc"]:
            alerts.append(f"ABS acc={acc:.3f} < {self.absolute['min_acc']}")
        if "max_ece" in self.absolute and e > self.absolute["max_ece"]:
            alerts.append(f"ABS ece={e:.3f} > {self.absolute['max_ece']}")
        if abs(acc - self.baseline_mean) > self.k_sigma * sd and self.baseline_mean is not None and len(self.log) > 30:
            alerts.append(f"REL acc={acc:.3f} beyond {self.k_sigma}sd of EWMA baseline")
        row["alerts"] = alerts
        self.log.append(row)
        return row


if __name__ == "__main__":
    print("=== Model monitoring: rolling metrics + EWMA baseline + alerts ===\n")
    rng = np.random.default_rng(0)
    K = 3
    T = 2000
    # Generate a stream where accuracy drops from ~0.90 to ~0.65 at t=1000.
    probs = np.zeros((T, K))
    y = rng.integers(0, K, T)
    for t in range(T):
        # Correct-class logit lower after t=1000
        logits = rng.normal(0, 0.5, K)
        logits[y[t]] += 2.5 if t < 1000 else 0.6
        probs[t] = _softmax(logits[None])[0]

    mon = Monitor(window=200, k_sigma=3.0,
                   alert_thresholds={"min_acc": 0.75, "max_ece": 0.30})
    first_abs, first_rel = None, None
    for t in range(T):
        row = mon.update(probs[t], y[t])
        if row and row["alerts"]:
            for a in row["alerts"]:
                if "ABS" in a and first_abs is None:
                    first_abs = t
                if "REL" in a and first_rel is None:
                    first_rel = t

    print(f"  true accuracy shift at t=1000")
    print(f"  first ABSOLUTE alert (acc < 0.75 or ece > 0.30): t = {first_abs}"
          f"   (latency = {first_abs - 1000 if first_abs else 'never'})")
    print(f"  first RELATIVE alert (3-sigma from EWMA):         t = {first_rel}"
          f"   (latency = {first_rel - 1000 if first_rel else 'never'})")

    # Report final rolling metrics
    last = mon.log[-1]
    print(f"\n  Final rolling metrics: acc={last['acc']:.3f}"
          f"   brier={last['brier']:.3f}"
          f"   ece={last['ece']:.3f}   baseline_acc={last['baseline_mean']:.3f}\n")
    print("--- library cross-check (evidently, whylogs, arize, seldon-alibi-detect) ---")
