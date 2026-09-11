"""SimPO - Simple Preference Optimization (Reference Sec 47.199).

Meng, Xia & Chen 2024 'SimPO: Simple Preference Optimization
with a Reference-Free Reward'. Like ORPO, SimPO removes the
reference model. The reward is LENGTH-NORMALISED average log
probability, and adds a TARGET MARGIN gamma to the pair loss:

    r(x, y) = (1 / |y|) * sum_t log pi(y_t | x, y_<t)     (avg log prob)
    L_SimPO = -log sigmoid(beta * (r(x, y_w) - r(x, y_l) - gamma))

Length normalisation cures DPO's LENGTH BIAS (chosen tends to be
shorter to raise total logp). The margin gamma tightens the
preference gap.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def dpo_loss(logp_chosen, logp_rejected, ref_c, ref_r, beta=0.1):
    """Standard DPO loss."""
    diff = beta * ((logp_chosen - ref_c) - (logp_rejected - ref_r))
    return -np.log(sigmoid(diff) + 1e-12)


def simpo_loss(logp_chosen, len_chosen, logp_rejected, len_rejected,
                 beta=2.0, gamma=0.5):
    """SimPO reference-free length-normalised loss."""
    r_c = logp_chosen / max(len_chosen, 1)
    r_r = logp_rejected / max(len_rejected, 1)
    diff = beta * (r_c - r_r - gamma)
    return -np.log(sigmoid(diff) + 1e-12)


if __name__ == "__main__":
    print("=== SimPO - Simple Preference Optimization (Meng et al 2024) ===\n")
    rng = np.random.default_rng(0)

    # 8 preference pairs; some 'chosen' happen to be much shorter than 'rejected'
    # DPO would exploit this length shortcut; SimPO corrects via length norm.
    print(f"  {'len_c':>7} {'len_r':>7} {'logp_c':>9} {'logp_r':>9}  "
          f"{'DPO loss':>10}  {'SimPO loss':>12}")
    total_dpo = 0; total_simpo = 0
    pairs = []
    for _ in range(8):
        len_c = int(rng.integers(3, 20))                        # chosen length
        len_r = int(rng.integers(5, 30))                        # rejected length (often longer)
        # Suppose per-token log-prob is roughly N(-0.5, 0.2) for both
        logp_c = -0.5 * len_c + rng.normal(scale=0.5)
        logp_r = -0.5 * len_r + rng.normal(scale=0.5) - 0.3      # slightly worse per-token
        pairs.append((len_c, len_r, logp_c, logp_r))
        ref_c = logp_c - 0.1; ref_r = logp_r - 0.1               # ref close to policy
        dpo_l = dpo_loss(logp_c, logp_r, ref_c, ref_r, beta=0.5)
        simpo_l = simpo_loss(logp_c, len_c, logp_r, len_r, beta=2.0, gamma=0.5)
        total_dpo += dpo_l; total_simpo += simpo_l
        print(f"  {len_c:>7d} {len_r:>7d} {logp_c:>9.2f} {logp_r:>9.2f}  "
              f"{dpo_l:>10.3f}  {simpo_l:>12.3f}")

    print(f"\n  Avg DPO   loss = {total_dpo / len(pairs):.3f}")
    print(f"  Avg SimPO loss = {total_simpo / len(pairs):.3f}")

    # Show length-bias effect: if chosen is much shorter, DPO rewards it disproportionately
    print("\n  DPO's total-logp is length-biased (shorter chosen gets more reward).")
    print("  SimPO's per-token log-prob puts long / short responses on equal footing.")

    print("\n--- library cross-check (trl.SimPOTrainer / axolotl SimPO config Python) ---")
