"""f-Divergences (Reference Sec 34.7).

Csiszar (1967).  Any convex f: (0, inf) -> R with f(1) = 0 defines

  D_f(p || q)  =  sum_x q(x) f( p(x) / q(x) ).

Special cases:
  f(u) = u log u                -> KL(p || q)
  f(u) = -log u                 -> reverse KL(q || p)  (via reversal)
  f(u) = 0.5 (u - 1)^2          -> Pearson chi-squared / 2
  f(u) = (sqrt(u) - 1)^2         -> squared Hellinger (times 2)
  f(u) = |u - 1| / 2             -> total variation
  f(u) = (1 - u^(1 - alpha)) / (alpha (1 - alpha))  -> alpha-divergence family
                                     alpha = 0.5 -> Hellinger; -> 0 or 1 -> KL branches.

Here we compute each on two discrete distributions + verify identities
(TV / Hellinger inequality, KL / chi^2 inequality).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kl(p, q, eps=1e-12):
    p, q = map(np.asarray, (p, q))
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))))


def chi2_pearson(p, q, eps=1e-12):
    return float(np.sum((np.asarray(p) - np.asarray(q)) ** 2 / (np.asarray(q) + eps)))


def hellinger_sq(p, q):
    return 0.5 * float(np.sum((np.sqrt(np.asarray(p)) - np.sqrt(np.asarray(q))) ** 2))


def total_variation(p, q):
    return 0.5 * float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def renyi(p, q, alpha, eps=1e-12):
    if abs(alpha - 1) < 1e-9:
        return kl(p, q)                       # alpha=1 limit
    p, q = map(np.asarray, (p, q))
    return float(1 / (alpha - 1) * np.log(np.sum(p ** alpha * (q + eps) ** (1 - alpha))))


if __name__ == "__main__":
    print("=== f-Divergences ===\n")
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    print("  p =", p.tolist(), "  q =", q.tolist())
    print(f"  KL(p || q)          = {kl(p, q):.4f} nats")
    print(f"  chi^2 (Pearson)     = {chi2_pearson(p, q):.4f}")
    print(f"  Hellinger^2         = {hellinger_sq(p, q):.4f}")
    print(f"  Total variation     = {total_variation(p, q):.4f}")
    print(f"  Renyi (alpha=2)     = {renyi(p, q, 2):.4f}")
    print(f"  Renyi (alpha=0.5)   = {renyi(p, q, 0.5):.4f}\n")

    # Inequality checks
    print("  Standard inequalities:")
    print(f"    TV <= sqrt(0.5 KL) ?      "
          f"{total_variation(p, q):.3f} <= {np.sqrt(0.5 * kl(p, q)):.3f}   (Pinsker)")
    print(f"    Hellinger^2 <= KL / 2 ?    "
          f"{hellinger_sq(p, q):.3f} <= {kl(p, q) / 2:.3f}\n")

    # Convergence: as q -> p, all divergences -> 0
    print("  As q -> p (small perturbation):")
    for eps in (0.1, 0.01, 0.001):
        q_e = 0.5 * q + 0.5 * p + eps * np.array([1, -1, 0])
        q_e = np.clip(q_e, 1e-6, 1)
        q_e = q_e / q_e.sum()
        print(f"    eps={eps:>6}   KL={kl(p, q_e):.4e}   chi^2={chi2_pearson(p, q_e):.4e}"
              f"   TV={total_variation(p, q_e):.4e}")

    print("\n--- library cross-check (scipy.stats + rel_entr; R philentropy; POT / GeomLoss) ---")
