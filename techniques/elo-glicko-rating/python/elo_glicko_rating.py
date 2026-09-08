"""Elo & Glicko rating systems (Reference Sec 47.66).

Elo 1978 'The Rating of Chessplayers, Past and Present'.
Deterministic online update after each game i vs j with result s
in {0, 0.5, 1}:

    E_ij = 1 / (1 + 10^((R_j - R_i) / 400))
    R_i  <- R_i + K * (s - E_ij)

K controls learning rate (32 in FIDE for players below 2400).

Glickman 1999 'Parameter estimation in large dynamic paired
comparison experiments'. Adds rating deviation RD (uncertainty)
and time-varying volatility. Bayesian rating conjugate updates.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def elo_update(R, i, j, s, K=32):
    """One Elo update; returns new (R_i, R_j)."""
    E_i = 1.0 / (1.0 + 10 ** ((R[j] - R[i]) / 400))
    delta = K * (s - E_i)
    R[i] += delta
    R[j] -= delta
    return R


def glicko_update(R, RD, opp_R, opp_RD, s, q=np.log(10) / 400):
    """Glicko batch update (Glickman 1999) for one player after 1 game.

    R, RD scalar (subject); opp_R, opp_RD arrays; s in {0, 0.5, 1}.
    """
    g = 1.0 / np.sqrt(1 + 3 * q * q * opp_RD ** 2 / np.pi ** 2)
    E = 1.0 / (1.0 + 10 ** (-g * (R - opp_R) / 400))
    d2_inv = q * q * np.sum(g * g * E * (1 - E))
    d2 = 1.0 / d2_inv
    R_new = R + (q / (1 / RD ** 2 + 1 / d2)) * np.sum(g * (s - E))
    RD_new = float(np.sqrt(1.0 / (1 / RD ** 2 + 1 / d2)))
    return R_new, RD_new


def simulate_elo(true_skill, n_games, K=32, base=1500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(true_skill)
    R = np.full(n, float(base))
    for _ in range(n_games):
        i, j = rng.choice(n, size=2, replace=False)
        p = 1.0 / (1 + np.exp(-(true_skill[i] - true_skill[j])))
        s = float(rng.random() < p)
        elo_update(R, i, j, s, K)
    return R


if __name__ == "__main__":
    print("=== Elo & Glicko rating systems (Elo 1978; Glickman 1999) ===\n")
    rng = np.random.default_rng(0)
    n_players = 8
    true_skill = np.linspace(-1.5, 1.5, n_players)     # log-odds skill
    R_final = simulate_elo(true_skill, n_games=4000, K=16, seed=0)
    print("  Elo (K=16, 4000 games):")
    print(f"    True log-odds skill: {np.round(true_skill, 2)}")
    print(f"    Final rating:        {np.round(R_final, 0)}")
    corr = np.corrcoef(true_skill, R_final)[0, 1]
    print(f"    Corr(rating, skill) = {corr:.3f}")

    # Glicko update demo: strong player beats 3 opponents
    R, RD = 1500.0, 350.0
    opp_R = np.array([1400, 1500, 1600], dtype=float)
    opp_RD = np.array([30, 100, 100], dtype=float)
    s = np.array([1.0, 1.0, 0.0])
    R_new, RD_new = glicko_update(R, RD, opp_R, opp_RD, s)
    print(f"\n  Glicko update:  R = 1500 (RD 350)  ->  R = {R_new:.1f}  (RD {RD_new:.1f})")

    print("\n--- library cross-check (PlayerRatings / EloRating R; skelo / trueskill Python) ---")
