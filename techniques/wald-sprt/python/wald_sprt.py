"""Wald Sequential Probability Ratio Test (Wald 1945, 1947).

For two simple hypotheses H0: theta=theta_0, H1: theta=theta_1,
accumulate the log-likelihood ratio:

    LR_n = sum log(f_theta1(x_i) / f_theta0(x_i))

Boundaries:
    log((1 - beta) / alpha)   -> reject H0  (accept H1)
    log(beta / (1 - alpha))   -> reject H1  (accept H0)
    otherwise -> keep sampling

Optimal average sample number among tests with (alpha, beta)
error rates (Wald-Wolfowitz optimality).
"""

import numpy as np    # arrays + random


def sprt(x, log_ratio_fn, alpha, beta):
    A = np.log((1 - beta) / alpha)
    B = np.log(beta / (1 - alpha))
    lr = 0.0
    for n, xn in enumerate(x, 1):
        lr = lr + log_ratio_fn(xn)
        if lr >= A:
            return "H1", n
        if lr <= B:
            return "H0", n
    return "no_decision", len(x)


def log_ratio_binomial(x, p0, p1):
    return np.log(p1 / p0) if x == 1 else np.log((1 - p1) / (1 - p0))


def demo():
    print("=== Wald SPRT (Wald 1945) ===")
    print("Binomial with H0: p=0.4  vs  H1: p=0.6  at alpha=beta=0.05")
    alpha, beta = 0.05, 0.05
    logA = np.log((1 - beta) / alpha)
    logB = np.log(beta / (1 - alpha))
    print(f"  Boundaries: log A = {logA:.3f}, log B = {logB:.3f}")

    rng = np.random.default_rng(2026)
    print("\nUnder H0 (true p=0.4), 5 000 trials:")
    ns_H0, decisions_H0 = [], []
    for trial in range(5000):
        x = rng.binomial(1, 0.4, size=2000)
        d, n = sprt(x, lambda z: log_ratio_binomial(z, 0.4, 0.6), alpha, beta)
        decisions_H0.append(d)
        ns_H0.append(n)
    rate_H1_given_H0 = sum(d == "H1" for d in decisions_H0) / len(decisions_H0)
    asn_H0 = np.mean(ns_H0)
    print(f"  Type-I  error (accept H1 | H0)  = {rate_H1_given_H0:.4f}  (target {alpha})")
    print(f"  ASN under H0                    = {asn_H0:.1f}")

    print("\nUnder H1 (true p=0.6), 5 000 trials:")
    ns_H1, decisions_H1 = [], []
    for trial in range(5000):
        x = rng.binomial(1, 0.6, size=2000)
        d, n = sprt(x, lambda z: log_ratio_binomial(z, 0.4, 0.6), alpha, beta)
        decisions_H1.append(d)
        ns_H1.append(n)
    rate_H0_given_H1 = sum(d == "H0" for d in decisions_H1) / len(decisions_H1)
    asn_H1 = np.mean(ns_H1)
    print(f"  Type-II error (accept H0 | H1)  = {rate_H0_given_H1:.4f}  (target {beta})")
    print(f"  ASN under H1                    = {asn_H1:.1f}")

    n_fixed = int((1.96 + 1.645) ** 2 * (0.4 * 0.6 + 0.6 * 0.4) / (0.6 - 0.4) ** 2)
    print(f"\nFor comparison, fixed-N sample size at same errors ~ {n_fixed} obs")


if __name__ == "__main__":
    demo()
