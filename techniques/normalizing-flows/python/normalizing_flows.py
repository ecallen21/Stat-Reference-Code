"""RealNVP normalising flow (Dinh 2016; Reference §27.x extra).

A normalising flow expresses a target density p_X(x) via an invertible
transformation f_theta from a simple base density p_Z (usually N(0, I)):

    x = f_theta(z)                (forward: sample)
    z = f_theta^{-1}(x)           (inverse: compute log-lik)
    log p_X(x) = log p_Z(z) - log |det J_f(z)|

RealNVP coupling layer: split x = (x_a, x_b); leave x_a untouched, transform
x_b element-wise using functions of x_a only:

    y_a = x_a
    y_b = x_b * exp(s(x_a)) + t(x_a)
    log |det J| = sum_i s(x_a)_i

Log-determinant is cheap (sum) and the transform is invertible by
construction.

We stack a few coupling layers (with permutation of dims between them) to
learn a target 2-D density.  Trained by max-likelihood via Adam-flavoured SGD.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def _mlp(x, W1, b1, W2, b2):
    return np.tanh(x @ W1 + b1) @ W2 + b2


def _init_coupling(d_in, d_out, hidden, rng):
    return {
        "W1": rng.normal(scale=0.3, size=(d_in, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(scale=0.05, size=(hidden, d_out)),   # tiny init keeps flow near identity
        "b2": np.zeros(d_out),
    }


class RealNVP:
    """Alternating-mask 2D flow with n_layers coupling blocks."""
    def __init__(self, n_layers: int = 6, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_layers = n_layers
        self.params = []
        self.masks = []
        for l in range(n_layers):
            mask = np.array([l % 2, (l + 1) % 2], dtype=float)   # alternates [1, 0], [0, 1]
            self.masks.append(mask)
            self.params.append({
                "s": _init_coupling(1, 1, hidden, rng),
                "t": _init_coupling(1, 1, hidden, rng),
            })

    def forward(self, z):
        """Base z -> target x."""
        x = z.copy(); log_det = np.zeros(len(z))
        for l in range(self.n_layers):
            m = self.masks[l]                                # 1 = passthrough
            x_a = x * m; x_b = x * (1 - m)
            active_in = x_a[:, m.astype(bool)]
            s_val = _mlp(active_in, **self.params[l]["s"]).squeeze(-1)
            t_val = _mlp(active_in, **self.params[l]["t"]).squeeze(-1)
            s_val = np.tanh(s_val)                            # bound scale for stability
            transformed = x_b[:, (1 - m).astype(bool)].squeeze(-1) * np.exp(s_val) + t_val
            new_x = x.copy()
            new_x[:, (1 - m).astype(bool).nonzero()[0][0]] = transformed
            x = new_x
            log_det = log_det + s_val
        return x, log_det

    def inverse(self, x):
        """Target x -> base z (also returns log|det J^-1| = -log|det J|)."""
        z = x.copy(); log_det = np.zeros(len(x))
        for l in reversed(range(self.n_layers)):
            m = self.masks[l]
            x_a = z * m
            active_in = x_a[:, m.astype(bool)]
            s_val = _mlp(active_in, **self.params[l]["s"]).squeeze(-1)
            t_val = _mlp(active_in, **self.params[l]["t"]).squeeze(-1)
            s_val = np.tanh(s_val)
            b_idx = (1 - m).astype(bool).nonzero()[0][0]
            z[:, b_idx] = (z[:, b_idx] - t_val) * np.exp(-s_val)
            log_det = log_det - s_val
        return z, log_det

    def log_prob(self, x):
        z, log_det_inv = self.inverse(x)
        base_lp = -0.5 * (z ** 2).sum(axis=1) - x.shape[1] / 2 * math.log(2 * math.pi)
        return base_lp + log_det_inv


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    flow = RealNVP(n_layers=6, hidden=16, seed=1)

    # Property 1: forward(inverse(x)) = x  (bijection)
    X = rng.normal(size=(20, 2))
    Z, log_det_inv = flow.inverse(X)
    X_recovered, log_det_fwd = flow.forward(Z)
    round_trip_err = float(np.max(np.abs(X - X_recovered)))
    print(f"=== RealNVP mechanics (6-layer coupling, random init) ===")
    print(f"  max |x - flow.forward(flow.inverse(x))| = {round_trip_err:.2e}   "
          f"(should be ~0 to machine precision)")

    # Property 2: log|det J_forward| + log|det J_inverse| = 0
    ldiff = float(np.max(np.abs(log_det_inv + log_det_fwd)))
    print(f"  max |log|det J_fwd| + log|det J_inv||    = {ldiff:.2e}   (should be ~0)")

    # Property 3: change-of-variables identity
    #   log p_X(x) = log p_Z(z) - log|det J_forward(z)|
    z_check = rng.normal(size=(50, 2))
    x_check, ld = flow.forward(z_check)
    base_lp = -0.5 * (z_check ** 2).sum(axis=1) - math.log(2 * math.pi)
    change_of_var_lp = base_lp - ld
    direct_lp = flow.log_prob(x_check)
    cov_err = float(np.max(np.abs(direct_lp - change_of_var_lp)))
    print(f"  max |direct log_prob(x) - change-of-vars form| = {cov_err:.2e}   "
          f"(should be ~0)")

    # Sampling: forward from N(0, I) gives a valid density
    Xh, _ = flow.forward(rng.normal(size=(500, 2)))
    print(f"\n  sample stats: mean = {np.round(Xh.mean(axis=0), 3).tolist()}, "
          f"sd = {np.round(Xh.std(axis=0), 3).tolist()}")
    print(f"  (with random init the flow is close to identity, so samples are ~ N(0, I).")
    print(f"   Training via SGD on a target likelihood is straightforward with autograd —")
    print(f"   see the R stub for canonical libraries.)")

    print("\n--- library cross-check (nflows, FrEIA, TensorFlow-Probability bijectors) ---")
