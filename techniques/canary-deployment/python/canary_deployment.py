"""Canary deployment (Reference Ch 32 MLOps).

Roll a candidate model into production by sending an INCREASING FRACTION
of traffic to it (5% -> 25% -> 50% -> 100%), monitoring performance +
error-budget metrics at each stage, and ROLLING BACK if a Service Level
Objective (SLO) is violated.

Typical rollout schedule:
  Stage 1:   5% traffic   for  1 hr    (bake-in)
  Stage 2:  25% traffic   for  1 hr
  Stage 3:  50% traffic   for  2 hr
  Stage 4: 100% traffic

At each stage, monitor:
  * ERROR RATE (or accuracy on the labelled subset).
  * LATENCY p99.
  * DOWNSTREAM business metric (revenue, conversion).

If any drift beyond an SLO threshold, roll back to previous stage.

Here we simulate a canary rollout in which the candidate degrades at
50% traffic (regression bug); the router auto-rolls-back and reports.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class CanaryRouter:
    def __init__(self, prod_fn, cand_fn, schedule, slo_error=0.20,
                  slo_latency_ms=200, seed=0):
        self.prod_fn, self.cand_fn = prod_fn, cand_fn
        self.schedule = schedule           # list of (traffic_fraction, n_requests)
        self.slo_error, self.slo_latency_ms = slo_error, slo_latency_ms
        self.rng = np.random.default_rng(seed)
        self.log = []
        self.current_stage = 0

    def _route(self, x, traffic):
        # Assign to candidate with probability = traffic.
        use_cand = self.rng.random() < traffic
        if use_cand:
            y_hat, latency = self.cand_fn(x)
            return "candidate", int(y_hat > 0.5), latency
        else:
            y_hat, latency = self.prod_fn(x)
            return "production", int(y_hat > 0.5), latency

    def run(self, X, y, max_rollbacks=2):
        stage = 0
        cursor = 0
        history = []
        rollbacks = 0
        while stage < len(self.schedule) and cursor < len(X):
            if rollbacks >= max_rollbacks:
                history.append({"stage": stage, "traffic": self.schedule[stage][0],
                                  "err_rate": float("nan"), "p99_ms": float("nan"),
                                  "cand_err_rate": float("nan"), "n": 0,
                                  "action": "HOLD_AT_STABLE (max rollbacks reached)"})
                break
            traffic, n_req = self.schedule[stage]
            # Simulate this stage.
            errs, lats, cand_errs, cand_n = 0, [], 0, 0
            for i in range(min(n_req, len(X) - cursor)):
                which, y_hat, lat = self._route(X[cursor], traffic)
                lats.append(lat)
                if y_hat != y[cursor]:
                    errs += 1
                    if which == "candidate":
                        cand_errs += 1
                if which == "candidate":
                    cand_n += 1
                cursor += 1
            err_rate = errs / max(n_req, 1)
            p99 = float(np.quantile(lats, 0.99))
            cand_err = cand_errs / max(cand_n, 1) if cand_n > 0 else 0.0
            history.append({"stage": stage, "traffic": traffic,
                             "err_rate": err_rate, "p99_ms": p99,
                             "cand_err_rate": cand_err, "n": n_req})
            slo_violation = err_rate > self.slo_error or p99 > self.slo_latency_ms
            if slo_violation and stage > 0:
                history[-1]["action"] = "ROLLBACK"
                stage -= 1
                rollbacks += 1
                continue
            elif slo_violation:
                history[-1]["action"] = "ABORT (cannot rollback below 0)"
                break
            else:
                history[-1]["action"] = "ADVANCE"
                stage += 1
        return history


if __name__ == "__main__":
    print("=== Canary deployment: progressive rollout with rollback ===\n")
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(0, 1, (n, 3))
    beta_true = np.array([1.5, -0.5, 0.7])
    y = ((1 / (1 + np.exp(-X @ beta_true))) > 0.5).astype(int)

    def prod_fn(x):
        # Stable production model
        prob = 1 / (1 + np.exp(-(x @ np.array([1.3, -0.4, 0.55]))))
        latency = float(rng.normal(80, 15))
        return prob, latency

    def cand_fn(x):
        # Candidate initially better but a bug injects noise when x[0] > 0.5.
        prob = 1 / (1 + np.exp(-(x @ np.array([1.5, -0.5, 0.7]))))
        if x[0] > 0.5:
            prob = 1 - prob                        # regression bug
        latency = float(rng.normal(90, 20))
        return prob, latency

    router = CanaryRouter(
        prod_fn=prod_fn, cand_fn=cand_fn,
        schedule=[(0.05, 400), (0.25, 400), (0.50, 400), (1.00, 400)],
        slo_error=0.20, slo_latency_ms=250, seed=0
    )
    history = router.run(X, y)

    print(f"  {'stage':>5}  {'traffic':>7}  {'err_rate':>8}  {'p99_ms':>7}"
          f"  {'cand_err':>8}  {'action':>10}")
    for h in history:
        print(f"  {h['stage']:>5}  {h['traffic']:>7.2f}  {h['err_rate']:>8.3f}"
              f"  {h['p99_ms']:>7.1f}  {h['cand_err_rate']:>8.3f}  {h['action']:>10}")

    print("\n  The router rolled back when the candidate regressed at higher traffic.\n")
    print("--- library cross-check (Istio VirtualService weighting; Argo Rollouts; Kubeflow) ---")
