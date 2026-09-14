"""Trust-region optimisation (Powell 1970; Steihaug 1983;
Byrd-Schnabel 2000).

At each iterate, solve the model sub-problem

    min_p m_k(p) = f_k + g_k^T p + 0.5 p^T H_k p
    subject to ||p|| <= delta_k

Update the trust-region radius delta_k based on actual/
predicted reduction ratio rho_k = (f(x) - f(x+p)) / (m(0) - m(p)).
Convergence guaranteed even from bad starting points.

This demo uses the DOG-LEG solver (Powell 1970) for the
sub-problem: Cauchy point + Newton step interpolation.
"""

import numpy as np    # arrays + linalg


def dog_leg(g, H, delta):
    """Solve trust-region sub-problem approximately by dog-leg."""
    pB = -np.linalg.solve(H + 1e-8 * np.eye(len(g)), g)
    if np.linalg.norm(pB) <= delta:
        return pB
    pU = -(g @ g) / (g @ H @ g) * g    # Cauchy
    if np.linalg.norm(pU) >= delta:
        return delta * pU / np.linalg.norm(pU)
    # solve ||pU + t (pB - pU)||^2 = delta^2
    d = pB - pU
    a = d @ d
    b = 2 * pU @ d
    c = pU @ pU - delta ** 2
    t = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return pU + t * d


def trust_region(f, grad, hess, x0, max_iter=100, delta0=1.0, eta=0.15):
    x = np.array(x0, dtype=float)
    delta = delta0
    history = [f(x)]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < 1e-8:
            break
        H = hess(x)
        p = dog_leg(g, H, delta)
        actual = f(x) - f(x + p)
        pred = -(g @ p + 0.5 * p @ H @ p)
        rho = actual / (pred + 1e-30)
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) > 0.99 * delta:
            delta = min(2 * delta, 10.0)
        if rho > eta:
            x = x + p
        history.append(f(x))
    return x, history


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
               for i in range(len(x) - 1))


def rosen_grad(x):
    g = np.zeros_like(x)
    for i in range(len(x) - 1):
        g[i] = g[i] - 400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] = g[i + 1] + 200 * (x[i + 1] - x[i] ** 2)
    return g


def rosen_hess(x):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] = H[i, i] + 1200 * x[i] ** 2 - 400 * x[i + 1] + 2
        H[i, i + 1] = H[i, i + 1] - 400 * x[i]
        H[i + 1, i] = H[i + 1, i] - 400 * x[i]
        H[i + 1, i + 1] = H[i + 1, i + 1] + 200
    return H


def demo():
    print("=== Trust-region optimisation (Powell 1970; dog-leg) ===")
    for dim in [2, 5, 10]:
        x0 = np.full(dim, -1.2)
        x_star, hist = trust_region(rosenbrock, rosen_grad, rosen_hess, x0,
                                     max_iter=100, delta0=1.0)
        print(f"  Rosenbrock d={dim}: iters = {len(hist) - 1}, f = {hist[-1]:.2e}, "
              f"‖x* - 1‖ = {np.linalg.norm(x_star - 1):.3e}")


if __name__ == "__main__":
    demo()
