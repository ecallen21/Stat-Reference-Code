"""Chambolle-Pock primal-dual algorithm (Chambolle-Pock 2011).

Solve saddle-point problem
    min_x max_y <K x, y> + f(x) - g*(y)

which arises in composite problems  min_x f(x) + g(K x)  via
Fenchel duality:

    y_new = prox_{sigma g*}(y + sigma K x_bar)
    x_new = prox_{tau f}(x - tau K.T y_new)
    x_bar = x_new + theta (x_new - x)     (extrapolation)

Convergence when  tau * sigma * ||K||^2 <= 1.  Ideal for TV
denoising and imaging inverse problems where K is a difference
operator.
"""

import numpy as np    # arrays + linalg


def chambolle_pock(prox_f, prox_g_star, K, K_T, x0, tau, sigma, theta=1.0,
                   max_iter=200):
    x = np.array(x0, dtype=float)
    x_bar = x.copy()
    y = np.zeros_like(K(x))
    for _ in range(max_iter):
        y = prox_g_star(y + sigma * K(x_bar), sigma)
        x_new = prox_f(x - tau * K_T(y), tau)
        x_bar = x_new + theta * (x_new - x)
        x = x_new
    return x


def demo():
    print("=== Chambolle-Pock primal-dual (Chambolle-Pock 2011) ===")
    print("TV denoising 1D: min 0.5 ||x - noisy||^2 + lam ||D x||_1")
    rng = np.random.default_rng(2026)
    n = 100
    # piecewise-constant signal + noise
    signal = np.zeros(n)
    signal[20:50] = 1.0
    signal[50:80] = -0.5
    noisy = signal + 0.3 * rng.standard_normal(n)

    # forward-difference operator D (n-1 x n)
    def K(x):
        return x[1:] - x[:-1]

    def K_T(y):
        z = np.zeros(len(y) + 1)
        z[:-1] = z[:-1] - y
        z[1:] = z[1:] + y
        return z

    lam = 0.5

    # f(x) = 0.5 ||x - noisy||^2  -> prox_{tau f}(z) = (z + tau noisy) / (1 + tau)
    def prox_f(z, tau):
        return (z + tau * noisy) / (1 + tau)

    # g(y) = lam ||y||_1  -> g*(w) = indicator [-lam, lam]
    # prox_{sigma g*}(w) = clip(w, -lam, lam)
    def prox_g_star(w, sigma):
        return np.clip(w, -lam, lam)

    tau = sigma = 0.4    # tau * sigma * 4 = 0.64 <= 1 (||D||^2 <= 4)
    x_denoised = chambolle_pock(prox_f, prox_g_star, K, K_T,
                                 x0=noisy.copy(), tau=tau, sigma=sigma,
                                 theta=1.0, max_iter=500)

    err_noisy = np.sqrt(np.mean((noisy - signal) ** 2))
    err_denoised = np.sqrt(np.mean((x_denoised - signal) ** 2))
    print(f"  n = {n}, noise sd = 0.3, lambda = {lam}")
    print(f"  Noisy    RMSE vs signal = {err_noisy:.4f}")
    print(f"  Denoised RMSE vs signal = {err_denoised:.4f}")
    print(f"  Number of remaining 'plateaus' (TV<0.01 pieces) = "
          f"{np.sum(np.abs(np.diff(x_denoised)) < 0.01)}")


if __name__ == "__main__":
    demo()
