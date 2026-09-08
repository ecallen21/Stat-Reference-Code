"""Welch's t-test (Reference Sec 47.113).

Welch 1947 'The generalization of "Student's" problem when several
different population variances are involved', Biometrika 34.
Two-sample t-test that does NOT assume equal variances:

    t = (xbar - ybar) / sqrt( s_x^2 / n_x + s_y^2 / n_y )
    df = (s_x^2/n_x + s_y^2/n_y)^2 /
          ((s_x^2/n_x)^2 / (n_x-1) + (s_y^2/n_y)^2 / (n_y-1))

Satterthwaite df. Preferred over Student's t in practice
(Delacre-Lakens-Leys 2017) even when variances look similar.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # t-distribution CDF


def welchs_t(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(sx2 / nx + sy2 / ny)
    t = (mx - my) / se
    df_num = (sx2 / nx + sy2 / ny) ** 2
    df_den = (sx2 / nx) ** 2 / (nx - 1) + (sy2 / ny) ** 2 / (ny - 1)
    df = df_num / df_den
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return {"t": float(t), "df": float(df), "p": float(p), "mean_diff": float(mx - my), "se": float(se)}


def students_t(x, y):
    """Classical pooled-variance t-test for comparison."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    se = np.sqrt(sp2 * (1 / nx + 1 / ny))
    t = (mx - my) / se
    df = nx + ny - 2
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return {"t": float(t), "df": float(df), "p": float(p)}


if __name__ == "__main__":
    print("=== Welch's t-test (Welch 1947) ===\n")
    rng = np.random.default_rng(0)

    scenarios = [
        ("equal sizes, equal variance",
         rng.normal(0.0, 1.0, 50), rng.normal(0.3, 1.0, 50)),
        ("equal sizes, unequal var (1, 3)",
         rng.normal(0.0, 1.0, 50), rng.normal(0.3, 3.0, 50)),
        ("unequal sizes (10, 200), equal var",
         rng.normal(0.0, 1.0, 10), rng.normal(0.3, 1.0, 200)),
        ("unequal sizes (10, 200), var (1, 5)",
         rng.normal(0.0, 1.0, 10), rng.normal(0.3, 5.0, 200)),
    ]
    print(f"  {'scenario':40s}  {'Welch p':>10s}  {'Student p':>10s}  {'Welch df':>10s}")
    for name, a, b in scenarios:
        w = welchs_t(a, b); s = students_t(a, b)
        print(f"  {name:40s}  {w['p']:10.4f}  {s['p']:10.4f}  {w['df']:10.1f}")

    # Type-I error simulation: unequal-var null, equal-var means -> Welch should hit ~0.05, Student inflated
    print("\n  Type-I error (5000 sims, mean diff = 0, sd = (1, 3), n = (20, 60)):")
    p_w, p_s = [], []
    for _ in range(5000):
        a = rng.normal(0.0, 1.0, 20)
        b = rng.normal(0.0, 3.0, 60)
        p_w.append(welchs_t(a, b)["p"]); p_s.append(students_t(a, b)["p"])
    print(f"    Welch's Type-I  rate = {np.mean(np.array(p_w) < 0.05):.4f}  (target 0.05)")
    print(f"    Student's Type-I rate = {np.mean(np.array(p_s) < 0.05):.4f}  (inflated when var unequal + unbalanced n)")

    print("\n--- library cross-check (stats::t.test(var.equal=FALSE) R; scipy.stats.ttest_ind(equal_var=False) Python) ---")
