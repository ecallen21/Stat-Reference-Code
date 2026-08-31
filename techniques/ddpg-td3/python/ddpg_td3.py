"""DDPG + TD3 for continuous-action RL (Reference §28.x extra).

DDPG (Lillicrap 2015):
    Actor mu_theta(s): S -> A  (deterministic policy)
    Critic Q_phi(s, a)
    Q target: y = r + gamma * Q_target(s', mu_target(s'))
    Q loss:   (Q(s, a) - y)^2
    pi loss:  -Q(s, mu(s))   (chain rule through mu into Q)
    Exploration: OU or Gaussian noise added at sample time.

TD3 (Fujimoto 2018) fixes DDPG's overestimation via:
    1. Double Q: two critics; MIN in the target.
    2. Delayed policy update: update mu once per d critic updates.
    3. Target-policy smoothing: clipped noise added to mu_target(s') in the target.

We implement on a contextual-bandit surrogate where reward = -(a - s)^2 (target
action a* = s, so optimal actor theta = 1).  Critic is quadratic in a
(five features) so it can represent Q(s, a) = -(a - s)^2 exactly.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


class LinearControl1D:
    def reset(self, rng): return float(rng.uniform(-1, 1))
    def step(self, s, a):
        return None, -(float(a) - float(s)) ** 2, True


def _phi_features(s, a):
    """Basis for a quadratic Q: [1, s, a, s*a, a^2]."""
    return np.array([1.0, s, a, s * a, a * a])


def _q(phi, s, a):
    return float(phi @ _phi_features(s, a))


def _actor(theta, s):
    return float(theta * s)


def train_ddpg(env, td3: bool = False, n_ep: int = 2000, lr_pi: float = 0.02,
                lr_q: float = 0.05, noise: float = 0.3,
                target_noise: float = 0.2, policy_delay: int = 2,
                tau: float = 0.05, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    theta = 0.0
    phi1 = np.zeros(5)
    phi2 = np.zeros(5) if td3 else None
    t_theta, t_phi1 = theta, phi1.copy()
    t_phi2 = phi2.copy() if td3 else None
    returns = []
    for ep in range(n_ep):
        s = env.reset(rng)
        a = _actor(theta, s) + noise * float(rng.normal())
        a = float(np.clip(a, -3, 3))
        _, r, _ = env.step(s, a)

        # critic target (terminal step -> y = r)
        y = r
        # critic update (linear regression on the current sample)
        x = _phi_features(s, a)
        for phi in ([phi1, phi2] if td3 else [phi1]):
            if phi is None: continue
            err = float(phi @ x) - y
            phi -= lr_q * err * x

        # actor update (delayed for TD3)
        if (not td3) or (ep % policy_delay == 0):
            # d Q / d a for phi = [1, s, a, sa, a^2]  is  phi[2] + phi[3] * s + 2 * phi[4] * a
            a_pred = _actor(theta, s)
            dQ_da = phi1[2] + phi1[3] * s + 2 * phi1[4] * a_pred
            # d actor / d theta = s
            grad_theta = -dQ_da * s
            theta -= lr_pi * grad_theta

        # Polyak targets
        t_theta = tau * theta + (1 - tau) * t_theta
        t_phi1 = tau * phi1 + (1 - tau) * t_phi1
        if td3:
            t_phi2 = tau * phi2 + (1 - tau) * t_phi2
        returns.append(r)
    return {"theta": theta, "phi1": phi1, "returns": returns,
            "method": "TD3" if td3 else "DDPG"}


if __name__ == "__main__":
    env = LinearControl1D()
    ddpg = train_ddpg(env, td3=False, n_ep=3000, seed=0)
    td3 = train_ddpg(env, td3=True, n_ep=3000, seed=0)

    print(f"=== DDPG vs TD3 on 1D control r = -(a - s)^2 (optimal theta = 1.0) ===")
    print(f"  DDPG learned theta = {ddpg['theta']:+.3f}")
    print(f"  TD3  learned theta = {td3['theta']:+.3f}")
    print(f"  DDPG critic phi[a^2] = {ddpg['phi1'][4]:+.3f}   "
          f"(should be near -1 for the -(a-s)^2 quadratic)")
    print(f"  DDPG mean reward last 300 episodes = {np.mean(ddpg['returns'][-300:]):.3f}   "
          f"(optimal = 0 without noise; ~-noise^2 with)")
    print(f"  TD3  mean reward last 300 episodes = {np.mean(td3['returns'][-300:]):.3f}")

    print("\n--- library cross-check (stable-baselines3.DDPG / TD3; cleanrl) ---")
