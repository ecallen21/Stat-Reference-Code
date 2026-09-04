# f-Divergences (Reference §34.7)

Csiszár (1967). Any convex `f: (0, ∞) → ℝ` with `f(1) = 0` defines

```
D_f(p ‖ q)  =  Σ_x q(x) f( p(x) / q(x) ).
```

## Special cases

| `f(u)`                                | Divergence           |
|--------------------------------------|----------------------|
| `u log u`                            | KL(p ‖ q)            |
| `− log u`                            | reverse-KL (via q, p)|
| `½ (u − 1)²`                         | Pearson χ² / 2       |
| `(√u − 1)²`                          | 2 × Hellinger²       |
| `½ |u − 1|`                          | total variation      |
| `(1 − u^(1−α)) / (α(1−α))`           | α-divergence / Rényi |

## Standard inequalities

- **Pinsker**: `TV ≤ √(½ KL)`.
- **Hellinger²** `≤ KL / 2`.
- **χ²** `≥ KL`.

## When to use

- **Choosing a distance for GANs / VI** — reverse KL, chi², Wasserstein.
- **Statistical distance in change detection** — TV is intuitive.
- **Kernel two-sample testing** — MMD / f-divergence bounds.
- **Robust statistics** — Hellinger distance in point estimation
  (Beran 1977).

## When NOT to use

- **Metric requirements** — most f-divergences are not metrics; use
  Wasserstein (`transport`) or squared Hellinger (which is a metric).
- **Non-overlapping support** — KL diverges; TV / Hellinger stay bounded.

## Files

- `python/f_divergences.py` — KL, Pearson χ², Hellinger², TV, Rényi.
  Demo on 3-cat distributions. **Standard inequalities verified**:
  Pinsker (`TV = 0.10 ≤ 0.112 = √(0.5·KL)`); Hellinger² ≤ KL/2.
  Convergence check as `q → p`.
- `r/f_divergences.R` — `philentropy`, `transport`, `FNN` (R);
  `scipy`, `POT` (Python).

## Assumptions & caveats

- **Support** — undefined where `q(x) = 0` but `p(x) > 0` for KL /
  Rényi (except α ∈ (0, 1)).
- **Sampling estimators** — k-NN KSG for continuous.
- **Non-metric** — asymmetric; not sub-additive.
- **Wasserstein** is an integral-probability metric, not an
  f-divergence (see `optimal-transport` if present).

## Related in this repo

- `kl-divergence`, `shannon-entropy`, `mutual-information`,
  `cross-entropy-log-loss` — sibling info-theoretic quantities.
- `data-drift-detection` — TV / Wasserstein appear as drift scores.
- `bootstrap`, `permutation-test` — resampling to test divergence
  significance.

## Run

```
python techniques/f-divergences/python/f_divergences.py
Rscript techniques/f-divergences/r/f_divergences.R
```

**Refs:** Csiszár, I. "Information-type measures of difference of probability distributions and indirect observations." *Studia Sci Math Hungarica*, 1967; Liese, F. & Vajda, I. "On divergences and informations in statistics and information theory." *IEEE Trans IT*, 2006.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
