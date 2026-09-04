"""Disproportionality analysis for pharmacovigilance signal detection (Reference Sec 43.1).

From spontaneous ADR reports (FAERS, VigiBase, EudraVigilance), build
a 2x2 for each (drug, event) pair:

                event        no-event
    drug         a              b
    no-drug      c              d

  PRR = (a / (a + b)) / (c / (c + d))               proportional reporting ratio
  ROR = (a / b) / (c / d)                            reporting odds ratio
  IC  = log2( (a + 0.5) / (E + 0.5) )                information component (Bayesian, BCPNN)
        where E = (a + b)(a + c) / (a + b + c + d)

Signal thresholds (EMA): PRR >= 2 AND a >= 3 AND chi^2 >= 4; IC025 > 0.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from math import log2


def prr(a, b, c, d):
    num = a / max(a + b, 1)
    den = c / max(c + d, 1)
    return float(num / max(den, 1e-12))


def ror(a, b, c, d):
    return float((a * d) / max(b * c, 1e-12))


def information_component(a, b, c, d):
    """BCPNN-style information component (Bate 1998)."""
    N = a + b + c + d
    E = (a + b) * (a + c) / N
    ic = log2((a + 0.5) / (E + 0.5))
    return {"IC": float(ic), "E": float(E)}


def signal_row(a, b, c, d):
    chi2 = 0.0
    N = a + b + c + d
    for obs, exp in [(a, (a + b) * (a + c) / N),
                     (b, (a + b) * (b + d) / N),
                     (c, (c + d) * (a + c) / N),
                     (d, (c + d) * (b + d) / N)]:
        chi2 += (obs - exp) ** 2 / max(exp, 1e-12)
    ic = information_component(a, b, c, d)
    return {"PRR": prr(a, b, c, d), "ROR": ror(a, b, c, d),
            "chi2": float(chi2), **ic}


if __name__ == "__main__":
    print("=== Disproportionality: PRR / ROR / IC / chi^2 signal detection ===\n")
    # Toy 2x2s
    cases = [
        ("drug1 -> event X (true signal)",  60,  400, 100, 20000),
        ("drug2 -> event Y (no signal)",     5,  500,  50, 20000),
        ("drug3 -> rare event",               3,  100,  20, 20000),
    ]
    for name, a, b, c, d in cases:
        r = signal_row(a, b, c, d)
        signal = (r["PRR"] >= 2 and a >= 3 and r["chi2"] >= 4)
        print(f"  {name}")
        print(f"    a = {a}, b = {b}, c = {c}, d = {d}")
        print(f"    PRR = {r['PRR']:.2f}   ROR = {r['ROR']:.2f}"
              f"   chi^2 = {r['chi2']:.1f}   IC = {r['IC']:+.3f}"
              f"   -> signal = {signal}")
        print()
    print("--- library cross-check (R PhViD::bcpnn/PRR, pvLRT; Python vigipy + custom) ---")
