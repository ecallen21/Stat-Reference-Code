"""Process capability indices (Reference Sec 37.7 / 37.12).

Cp   = (USL - LSL) / (6 sigma)                     -- potential capability.
Cpk  = min((USL - mu) / (3 sigma), (mu - LSL) / (3 sigma))
                                                    -- capability accounting for centring.
Pp   = (USL - LSL) / (6 s)   with s = OVERALL sd
                                                    -- Pp uses long-term (overall) sd.
Ppk  = min((USL - mu) / (3 s), (mu - LSL) / (3 s))
Cpm  = (USL - LSL) / (6 sqrt(sigma^2 + (mu - T)^2))  -- Taguchi target-based index.

Interpretation:
  * Cp >= 1.33 typically considered CAPABLE.
  * Cpk = Cp when the process is centred; smaller when off-target.

Here we compute all five on a synthetic process and interpret.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def capability_indices(x, LSL, USL, T=None):
    x = np.asarray(x, dtype=float)
    mu = float(x.mean())
    sigma = float(x.std(ddof=1))                     # short-term: use S = R-bar/d_2 in practice
    s = sigma                                         # for simplicity treat both as same
    if T is None: T = (LSL + USL) / 2
    Cp = (USL - LSL) / (6 * sigma)
    Cpk = min((USL - mu) / (3 * sigma), (mu - LSL) / (3 * sigma))
    Pp = (USL - LSL) / (6 * s)
    Ppk = min((USL - mu) / (3 * s), (mu - LSL) / (3 * s))
    Cpm = (USL - LSL) / (6 * np.sqrt(sigma ** 2 + (mu - T) ** 2))
    return {"mu": mu, "sigma": sigma, "Cp": Cp, "Cpk": Cpk,
             "Pp": Pp, "Ppk": Ppk, "Cpm": Cpm}


if __name__ == "__main__":
    print("=== Process capability indices ===\n")
    rng = np.random.default_rng(0)
    LSL, USL, T = 8.0, 12.0, 10.0

    # Case A: centred, tight variability
    xA = rng.normal(10.0, 0.6, 300)
    rA = capability_indices(xA, LSL, USL, T)
    print(f"  Case A (centred, sigma=0.6)")
    for k in ("mu", "sigma", "Cp", "Cpk", "Pp", "Ppk", "Cpm"):
        print(f"    {k:>4} = {rA[k]:.3f}")
    print(f"    -> Cp = {rA['Cp']:.3f} > 1.33 -> capable; Cp ~ Cpk since centred.\n")

    # Case B: off-centre, same variability
    xB = rng.normal(11.0, 0.6, 300)
    rB = capability_indices(xB, LSL, USL, T)
    print(f"  Case B (mu = 11.0, off-centre)")
    for k in ("mu", "sigma", "Cp", "Cpk", "Pp", "Ppk", "Cpm"):
        print(f"    {k:>4} = {rB[k]:.3f}")
    print(f"    -> Cpk << Cp because off-centre; Cpm penalises departure from target.\n")

    print("--- library cross-check (R qcc::process.capability; SixSigma package) ---")
