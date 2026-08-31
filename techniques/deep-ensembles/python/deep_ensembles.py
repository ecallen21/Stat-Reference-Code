"""Deep ensembles (Reference Ch 29 Uncertainty Quantification).

Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles."

Train K neural nets from DIFFERENT random initialisations on the SAME data;
at test time average the predictive DISTRIBUTIONS (not just point estimates).
The disagreement across members is a plug-in estimate of EPISTEMIC uncertainty
and predictive intervals widen naturally in extrapolation regions.

  mu(x)      = (1/K) sum_k mu_k(x)
  var(x)     = (1/K) sum_k (sigma_k(x)^2 + mu_k(x)^2) - mu(x)^2
                          <- aleatoric ->        <- epistemic ->

Each member outputs (mu_k, sigma_k^2) so we get aleatoric AND epistemic in
one shot. Here we implement a MINI feed-forward regressor with the Gaussian
NLL head from Nix-Weigend / Lakshminarayanan.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(x): return np.maximum(x, 0.0)


def _init_mlp(rng, d_in=1, d_hid=32, d_out=2):
    # d_out = 2 -> (mu, log_var)
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hid)),
        "b1": np.zeros(d_hid),
        "W2": rng.normal(0, np.sqrt(2.0 / d_hid), (d_hid, d_out)),
        "b2": np.zeros(d_out),
    }


def _forward(p, x):
    h = _relu(x @ p["W1"] + p["b1"])
    out = h @ p["W2"] + p["b2"]
    mu = out[:, 0]
    log_var = np.clip(out[:, 1], -6.0, 6.0)
    return mu, np.exp(log_var), h


def _gauss_nll(mu, var, y):
    # 0.5 log var + (y-mu)^2 / (2 var), summed
    return 0.5 * np.log(var) + 0.5 * (y - mu) ** 2 / var


def _train_member(x, y, rng, d_hid=32, lr=1e-2, epochs=2000):
    p = _init_mlp(rng, x.shape[1], d_hid, 2)
    n = x.shape[0]
    for _ in range(epochs):
        mu, var, h = _forward(p, x)
        d_mu = -(y - mu) / var / n
        d_lv = 0.5 * (1.0 - (y - mu) ** 2 / var) / n
        d_out = np.stack([d_mu, d_lv], axis=1)
        d_W2 = h.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h = d_out @ p["W2"].T
        d_h[h <= 0] = 0.0
        d_W1 = x.T @ d_h
        d_b1 = d_h.sum(axis=0)
        p["W1"] -= lr * d_W1; p["b1"] -= lr * d_b1
        p["W2"] -= lr * d_W2; p["b2"] -= lr * d_b2
    return p


def deep_ensemble(x, y, K=5, d_hid=32, seed=0):
    rng = np.random.default_rng(seed)
    members = []
    for k in range(K):
        sub_rng = np.random.default_rng(seed + 17 * k + 1)
        members.append(_train_member(x, y, sub_rng, d_hid=d_hid))
    return members


def ensemble_predict(members, x):
    mus, vars_ = [], []
    for p in members:
        m, v, _ = _forward(p, x)
        mus.append(m); vars_.append(v)
    mus = np.array(mus)       # (K, N)
    vars_ = np.array(vars_)   # (K, N)
    mu = mus.mean(axis=0)
    aleatoric = vars_.mean(axis=0)                     # mean of variances
    epistemic = mus.var(axis=0)                        # variance of means
    var_total = aleatoric + epistemic
    return {"mu": mu, "var": var_total,
            "aleatoric": aleatoric, "epistemic": epistemic}


if __name__ == "__main__":
    print("=== Deep ensembles (Lakshminarayanan 2017) ===\n")
    rng = np.random.default_rng(0)
    # y = sin(2x) on [-2, 2] with heteroscedastic noise; test on [-3, 3]
    x_tr = rng.uniform(-2, 2, 80).reshape(-1, 1)
    noise = 0.05 + 0.15 * np.abs(x_tr[:, 0])
    y_tr = np.sin(2 * x_tr[:, 0]) + rng.normal(0, noise)

    members = deep_ensemble(x_tr, y_tr, K=5)
    x_te = np.linspace(-3, 3, 21).reshape(-1, 1)
    r = ensemble_predict(members, x_te)

    print(f"  {'x':>6}  {'mu':>7}  {'sd':>6}  {'aleat':>7}  {'epist':>7}  region")
    for i, xv in enumerate(x_te[:, 0]):
        region = "in " if -2 <= xv <= 2 else "out"
        sd = np.sqrt(r["var"][i])
        print(f"  {xv:>6.2f}  {r['mu'][i]:>7.3f}  {sd:>6.3f}"
              f"  {np.sqrt(r['aleatoric'][i]):>7.3f}"
              f"  {np.sqrt(r['epistemic'][i]):>7.3f}  {region}")

    in_mask = (x_te[:, 0] >= -2) & (x_te[:, 0] <= 2)
    out_mask = ~in_mask
    print(f"\n  mean epistemic sd  in-support: {np.sqrt(r['epistemic'][in_mask]).mean():.3f}")
    print(f"  mean epistemic sd out-support: {np.sqrt(r['epistemic'][out_mask]).mean():.3f}")
    ratio = np.sqrt(r['epistemic'][out_mask]).mean() / np.sqrt(r['epistemic'][in_mask]).mean()
    print(f"  ratio (out/in): {ratio:.2f}x  <- epistemic should be larger outside training support\n")

    print("--- library cross-check (torch nn.Module + Nix-Weigend head; skorch NLL) ---")
