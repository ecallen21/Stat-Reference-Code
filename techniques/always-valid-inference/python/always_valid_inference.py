"""Always-valid p-values (Reference Sec 44.4, 44.12).

Johari-Koomen-Pekelis-Walsh 2017: continuous monitoring of A/B
tests inflates type-I error under fixed-horizon methods.  Two
peek-safe alternatives:

  mSPRT (mixture SPRT):
    Test H0: mu = 0 vs H1: mu != 0 (with a mixing prior on effect
    size).  Yields an always-valid p_t at every time t so that
    P(any t: p_t <= alpha under H0) <= alpha.

  CONFIDENCE SEQUENCES (Howard-Ramdas-McAuliffe-Sekhon 2021):
    Time-uniform CI: P(mu in CS_t for all t) >= 1 - alpha.

Compact demo: mSPRT for a Gaussian effect + confidence sequence.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def msprt_p_gaussian(delta_hat_seq, se_seq, tau2=1.0):
    """Robbins mixture SPRT p-value for a running z-score sequence.

    Under H0: cumulative z ~ N(0, 1).  Mixture over effect ~ N(0, tau2)
    gives an always-valid p = 1 / mixture_LR.
    """
    p_seq = []
    for z_t, n_t in zip(delta_hat_seq, se_seq):
        # LR of N(0, 1 + n * tau2) vs N(0, 1)  evaluated at z_t
        var = 1 + n_t * tau2
        log_mix = -0.5 * np.log(var) + 0.5 * z_t ** 2 * (1 - 1 / var)
        p_seq.append(min(1.0, float(np.exp(-log_mix))))
    return p_seq


def confidence_sequence_gaussian(mean_seq, se_seq, alpha=0.05):
    """Howard-Ramdas time-uniform Gaussian confidence sequence.

    CS_t = mean_hat_t +/- se_t * sqrt( 2 * (log(1/alpha) + 0.5 * log(1 + t)) )
    """
    return [se_t * np.sqrt(2 * (np.log(1 / alpha) + 0.5 * np.log(1 + t + 1)))
            for t, se_t in enumerate(se_seq)]


if __name__ == "__main__":
    print("=== Always-valid inference: mSPRT + confidence sequence ===\n")
    rng = np.random.default_rng(0)
    # Simulate a "peek every batch" A/B under H0 and H1
    for name, effect in [("H0 (no effect)", 0.0), ("H1 (small effect)", 0.10)]:
        obs = rng.normal(effect, 1.0, 2000)
        cum_mean = np.cumsum(obs) / (np.arange(1, len(obs) + 1))
        n = np.arange(1, len(obs) + 1)
        se = 1.0 / np.sqrt(n)
        z = cum_mean / se
        p_msprt = msprt_p_gaussian(z, n, tau2=0.5)
        cs_half = confidence_sequence_gaussian(cum_mean, se)
        # Naive fixed-horizon p at every t
        from scipy.stats import norm
        p_naive = 2 * norm.sf(np.abs(z))
        # Ratio of times each rejected at alpha=0.05
        rej_naive = float((p_naive <= 0.05).sum() / len(p_naive))
        rej_msprt = float(sum(1 for p in p_msprt if p <= 0.05) / len(p_msprt))
        # Did the CS EVER exclude 0?
        never_included_0 = any(abs(m) > h for m, h in zip(cum_mean, cs_half))
        print(f"  {name}")
        print(f"    naive p<=0.05 rate over 2000 peeks : {rej_naive:.3f}")
        print(f"    mSPRT p<=0.05 rate over 2000 peeks : {rej_msprt:.3f}")
        print(f"    CS ever excluded 0 : {never_included_0}\n")

    print("--- library cross-check (R gsDesign/rpact/ldbounds; Python confidence-sequence, custom) ---")
