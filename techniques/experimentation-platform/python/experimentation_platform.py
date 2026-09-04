"""Experimentation platform primitives (Reference Sec 44.11).

Kohavi-Tang-Xu 2020 Ch 3-5, 22.  Production experimentation
infrastructure needs:

  * Deterministic RANDOMISATION UNIT -> variant hashing.
  * NAMESPACES / LAYERS so concurrent experiments don't collide
    (Bakshy et al. Facebook PlanOut).
  * SAMPLE RATIO MISMATCH (SRM) check: chi^2 vs expected split.
  * INTERACTION detection across experiments.

Compact demo: hash-based assignment + namespace, SRM check.
"""
from __future__ import annotations    # stdlib

import hashlib
from typing import Sequence

import numpy as np    # numerical arrays
from scipy import stats


def assign(user_id, experiment_name, weights=(0.5, 0.5)):
    """Deterministic user -> variant hash within an experiment namespace."""
    h = int(hashlib.md5(f"{experiment_name}:{user_id}".encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if r < cum:
            return i
    return len(weights) - 1


def srm_check(counts, expected_weights):
    """Chi^2 SRM check: does actual traffic split match design?"""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    exp = n * np.asarray(expected_weights, dtype=float)
    chi2 = ((counts - exp) ** 2 / exp).sum()
    dof = len(counts) - 1
    p = 1 - stats.chi2.cdf(chi2, df=dof)
    return {"chi2": float(chi2), "p_value": float(p), "expected": exp.tolist(),
            "observed": counts.tolist(),
            "srm_detected": bool(p < 0.001)}


if __name__ == "__main__":
    print("=== Experimentation platform: hash assignment + SRM check ===\n")
    n_users = 100000
    counts = [0, 0]
    for uid in range(n_users):
        counts[assign(uid, "checkout_v3", weights=(0.5, 0.5))] += 1
    r = srm_check(counts, expected_weights=(0.5, 0.5))
    print(f"  Traffic split (50/50 planned) : {counts}")
    print(f"  SRM chi^2 = {r['chi2']:.2f}, p = {r['p_value']:.3f}"
          f"   -> SRM detected = {r['srm_detected']}")

    # A different namespace: independent assignment
    print("\n  Different experiment namespace decouples assignments:")
    same_uid = 12345
    print(f"    user {same_uid} in checkout_v3      -> variant {assign(same_uid, 'checkout_v3')}")
    print(f"    user {same_uid} in header_color     -> variant {assign(same_uid, 'header_color')}")
    print(f"    user {same_uid} in recommender_algo -> variant {assign(same_uid, 'recommender_algo')}")

    # Buggy assignment (bad rand seed) creates SRM
    counts_bad = [45000, 55000]
    r_bad = srm_check(counts_bad, (0.5, 0.5))
    print(f"\n  Bad-assignment case: {counts_bad}   chi^2 = {r_bad['chi2']:.1f}"
          f"   p = {r_bad['p_value']:.2e}   -> SRM detected = {r_bad['srm_detected']}\n")

    print("--- library cross-check (R stats::chisq.test, custom; Python planout, scipy) ---")
