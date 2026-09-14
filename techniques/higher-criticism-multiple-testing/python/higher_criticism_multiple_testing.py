"""Higher Criticism (Donoho & Jin 2004 Ann Statist).

For n p-values sorted p_(1) <= p_(2) <= ... <= p_(n), form

    HC_i = sqrt(n) * (i/n - p_(i)) / sqrt(p_(i) * (1 - p_(i)))

Higher-Criticism statistic: HC* = max_{i: alpha0 <= i/n <= alpha0'} HC_i.
Optimal detection boundary for very sparse mixtures of signals
against null (asymptotically minimax in Ingster 1997 detection
regime).
"""

import numpy as np    # arrays + stats


def higher_criticism(p_values, alpha0=0.005, alpha_end=0.5):
    n = len(p_values)
    p_sorted = np.sort(p_values)
    i_grid = np.arange(1, n + 1)
    valid = (i_grid >= alpha0 * n) & (i_grid <= alpha_end * n)
    i_valid = i_grid[valid]
    p_valid = p_sorted[valid]
    hc_i = np.sqrt(n) * (i_valid / n - p_valid) / np.sqrt(p_valid * (1 - p_valid) + 1e-30)
    hc_star = np.max(hc_i)
    i_argmax = i_valid[np.argmax(hc_i)]
    return hc_star, i_argmax, hc_i, i_valid


def demo():
    print("=== Higher Criticism (Donoho-Jin 2004 Ann Statist) ===")
    rng = np.random.default_rng(2026)

    # Setup: n hypotheses, most null (z ~ N(0, 1)), rare signals (z ~ N(mu, 1))
    for n, frac_signal, mu in [(1000, 0.01, 3.5),
                                (1000, 0.05, 3.0),
                                (1000, 0.001, 5.0)]:
        n_signal = int(n * frac_signal)
        z = rng.standard_normal(n)
        z[:n_signal] = z[:n_signal] + mu
        rng.shuffle(z)
        p = 2 * (1 - _norm_cdf(np.abs(z)))

        hc_star, i_star, _, _ = higher_criticism(p)
        # global null threshold ~ sqrt(2 * log(log(n))) * (1 + o(1))
        crit = np.sqrt(2 * np.log(np.log(n)))
        rej = hc_star > crit
        print(f"  n = {n}, signal fraction = {frac_signal:.3f}, mu = {mu}: "
              f"HC* = {hc_star:.2f} vs threshold {crit:.2f} => "
              f"{'REJECT global null' if rej else 'accept'}")


def _norm_cdf(x):
    """CDF of standard normal (avoid scipy dependency)."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x):
    """Abramowitz-Stegun rational approximation for erf."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


if __name__ == "__main__":
    demo()
