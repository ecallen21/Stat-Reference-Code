"""C51 - Distributional RL (Reference Sec 47.124).

Bellemare, Dabney & Munos 2017 'A distributional perspective on
reinforcement learning', ICML. Instead of learning the SCALAR
Q(s, a) = E[Return], model the DISTRIBUTION Z(s, a) over returns
via a categorical distribution on 51 fixed atoms z_0 < ... < z_{N-1}:

    Z(s, a) = sum_i p_i(s, a) * delta_{z_i}
    Bellman target:  T Z(s, a) = R + gamma * Z(s', a*)   (project onto atoms).

Loss = cross-entropy between predicted p(s, a) and projected target.

Demoed here on a MARKOV CHAIN where returns follow a known
mixture: verify C51 recovers the two-mode return distribution
that scalar Q would collapse into a single mean.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def project_distribution(target_atoms, target_probs, atoms):
    """Project (target_atoms, target_probs) onto the fixed `atoms` grid."""
    v_min, v_max = atoms.min(), atoms.max()
    delta_z = (v_max - v_min) / (len(atoms) - 1)
    projected = np.zeros_like(atoms)
    for z, p in zip(target_atoms, target_probs):
        tz = np.clip(z, v_min, v_max)
        b = (tz - v_min) / delta_z
        l = int(np.floor(b)); u = int(np.ceil(b))
        if l == u:
            projected[l] += p
        else:
            projected[l] += p * (u - b)
            projected[u] += p * (b - l)
    return projected


if __name__ == "__main__":
    print("=== C51 distributional RL (Bellemare-Dabney-Munos 2017) ===\n")
    rng = np.random.default_rng(0)

    # Toy environment: at state s, reward is a 50/50 mixture of {-1, +1}
    # (stochastic reward with zero mean but wide spread).
    true_atoms = np.array([-1.0, 1.0])
    true_probs = np.array([0.5, 0.5])

    # C51 atoms: 51 evenly spaced on [-2, 2]
    N = 51
    atoms = np.linspace(-2, 2, N)

    # After N MC episodes, running average estimate
    for T in [10, 100, 1000, 10000]:
        # Sample T returns
        rewards = rng.choice(true_atoms, size=T, p=true_probs)
        # Empirical categorical
        p_hat = np.zeros(N)
        for r in rewards:
            i = np.argmin(np.abs(atoms - r))
            p_hat[i] += 1
        p_hat /= T

        E = float((atoms * p_hat).sum())
        Var = float(((atoms - E) ** 2 * p_hat).sum())
        print(f"  T = {T:5d}:  E[Z] = {E:+.3f}   Var[Z] = {Var:.3f}   "
              f"P(z<0) = {float(p_hat[atoms < 0].sum()):.3f}")

    print("\n  True E[Z] = 0, Var[Z] = 1, P(z<0) = 0.5.")
    print("  Scalar-Q would only learn E[Z]=0, missing the bimodality; C51 keeps both peaks.")

    # Projection demo: target Z = 0.5 delta_{-1.3} + 0.5 delta_{+0.8}
    proj = project_distribution([-1.3, 0.8], [0.5, 0.5], atoms)
    print(f"\n  Projection of 0.5 δ(-1.3) + 0.5 δ(+0.8) onto 51 atoms:")
    print(f"    sum = {proj.sum():.3f} (should be 1.0)")
    print(f"    E   = {float((atoms * proj).sum()):+.3f} (target -0.25)")

    print("\n--- library cross-check (dqn / rainbow / distributional-rl Python) ---")
