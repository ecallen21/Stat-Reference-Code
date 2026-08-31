"""Concept-drift detection: ADWIN + DDM (Reference Ch 32 MLOps).

  ADWIN (Bifet-Gavalda 2007) 'Learning from Time-Changing Data with
    Adaptive Windowing.'
  DDM   (Gama 2004) 'Learning with Drift Detection.'

Both are ONLINE, single-pass drift detectors on a stream of binary
losses / errors.

ADWIN maintains a variable-length window W of recent observations.
For every pair of contiguous sub-windows (W = W0 || W1), it tests

  |mean(W0) - mean(W1)|  >  epsilon(delta, n0, n1)

where the Hoeffding-style bound epsilon guarantees FALSE-ALARM rate
<= delta.  If the test fires, the older sub-window W0 is DROPPED
(concept has changed there); W1 becomes the new window.

DDM tracks the running error rate p_i and its std sigma_i = sqrt(p*(1-p)/i).
Two thresholds on p + sigma:

  WARNING zone when   p + sigma > p_min + 2 sigma_min.
  DRIFT   flag when   p + sigma > p_min + 3 sigma_min.

Here we implement both on a simulated stream where the label-conditional
distribution shifts at t = 500, and report the drift-detection latency.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


class ADWIN:
    def __init__(self, delta=0.002, max_buckets=5):
        self.delta = delta
        self.window = []                # list of 0/1 losses
        self.drift_at = []              # indices where drift was flagged

    def _eps(self, n0, n1):
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        return float(np.sqrt(2.0 / m * np.log(2.0 / self.delta)))

    def update(self, x, t):
        self.window.append(int(x))
        drift = False
        # Test every split point (light version, O(n) per update).
        for split in range(1, len(self.window)):
            W0 = self.window[:split]
            W1 = self.window[split:]
            if abs(np.mean(W0) - np.mean(W1)) > self._eps(len(W0), len(W1)):
                # Drop old sub-window.
                self.window = W1
                self.drift_at.append(t)
                drift = True
                break
        return drift


class DDM:
    def __init__(self):
        self.n = 0
        self.p = 0.0
        self.sigma = 0.0
        self.p_min = float("inf")
        self.sigma_min = float("inf")
        self.state = "normal"

    def update(self, error):
        self.n += 1
        # Streaming mean of error
        self.p += (int(error) - self.p) / self.n
        self.sigma = np.sqrt(self.p * (1 - self.p) / max(self.n, 1))
        if self.n < 30:
            return self.state
        if self.p + self.sigma < self.p_min + self.sigma_min:
            self.p_min = self.p
            self.sigma_min = self.sigma
        if self.p + self.sigma > self.p_min + 3 * self.sigma_min:
            self.state = "drift"
        elif self.p + self.sigma > self.p_min + 2 * self.sigma_min:
            self.state = "warning"
        else:
            self.state = "normal"
        return self.state


if __name__ == "__main__":
    print("=== ADWIN + DDM online concept-drift detection ===\n")
    rng = np.random.default_rng(0)
    # Stream of 1000 loss indicators.
    # Before t=500: error prob 0.10 (concept A).
    # After  t=500: error prob 0.40 (concept B).
    T = 1000
    errors = np.concatenate([(rng.random(500) < 0.10).astype(int),
                              (rng.random(500) < 0.40).astype(int)])

    adwin = ADWIN(delta=0.002)
    ddm = DDM()
    first_adwin, first_ddm_warn, first_ddm_drift = None, None, None
    ddm_states = []
    for t, e in enumerate(errors):
        d = adwin.update(e, t)
        if d and first_adwin is None:
            first_adwin = t
        state = ddm.update(e)
        ddm_states.append(state)
        if state == "warning" and first_ddm_warn is None:
            first_ddm_warn = t
        if state == "drift" and first_ddm_drift is None:
            first_ddm_drift = t

    print(f"  true drift point:     t = 500")
    print(f"  ADWIN first flag:    t = {first_adwin}   (latency = {first_adwin - 500 if first_adwin else 'never'})")
    def _lat(x):
        if x is None: return "never"
        return f"{x - 500}" if x >= 500 else f"early false-alarm at {x}"
    print(f"  DDM  first warning:  t = {first_ddm_warn} ({_lat(first_ddm_warn)})")
    print(f"  DDM  first drift:    t = {first_ddm_drift} ({_lat(first_ddm_drift)})")

    print(f"\n  Total ADWIN drift flags in the run: {len(adwin.drift_at)}"
          f"   (flags at {adwin.drift_at[:5]}{'...' if len(adwin.drift_at) > 5 else ''})\n")
    print("--- library cross-check (river.drift.ADWIN / DDM; skmultiflow.drift_detection) ---")
