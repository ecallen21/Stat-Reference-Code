# Rademacher Complexity (Reference §46.10)

Bartlett & Mendelson (2002). Distribution-dependent measure of a
function class `F`'s ability to fit random labels. Drives
finite-sample generalisation bounds in learning theory.

## Definition

    R̂_n(F) = E_σ [ sup_{f ∈ F} (1/n) Σᵢ σᵢ · f(xᵢ) ]

with `σᵢ ∈ {−1, +1}` iid Rademacher (uniform) noise.

## Bartlett-Mendelson theorem

With probability ≥ 1 − δ over the sample, for every `f ∈ F`:

    L(f) ≤ L̂_n(f) + 2 · R̂_n(F) + 3 · √(log(2/δ) / (2n))

The `2·R̂_n(F)` term is the **complexity price** for the hypothesis
search. Smaller R̂_n ⇒ better generalisation.

## Files

- `python/rademacher_complexity.py` — Monte-Carlo estimator for two
  classes: linear predictors `{⟨w, x⟩ : ‖w‖ ≤ B}` (closed-form
  `sup`) and depth-1 decision stumps (search over cuts).
  Demo (n=200, p=5, unit rows): R̂ = 0.066 for `‖w‖ ≤ 1` matches
  the theoretical `B/√n = 0.071`; scales linearly with B (0.20 at
  B=3); stumps are richer (0.17).
- `r/rademacher_complexity.R` — no CRAN package; implement from
  scratch (R stub explains).

## Known closed forms

| Class | R̂_n(F) upper bound |
|---|---|
| Linear ball ‖w‖ ≤ B | `B · max_i ‖xᵢ‖ / √n` |
| Finite class | `√(2 log|F| / n)` (Massart) |
| VC-dim d | `O(√(d log(n) / n))` (Dudley) |
| Kernel k | `√(tr(K)) / n · B` |

## When to use

- **Theoretical bounds** — proving generalisation for a class.
- **Model comparison** — richer class ⇒ larger R̂; use to justify
  regularisation strength.
- **Choosing complexity penalties** — Rademacher-based structural
  risk minimisation.

## When NOT to use

- **Practical tuning** — CV / hold-out gives tighter empirical error
  estimates. Rademacher is best for theory.
- **Non-iid or streaming data** — the definition needs iid; use
  sequential-Rademacher variants.

## Assumptions & caveats

- **Bounded losses / functions** — the bounds assume `|f| ≤ M`.
- **Estimator variance** — the `E_σ` outer expectation is estimated
  by Monte Carlo; use many draws for stable estimates.
- **Gap to true risk** — the bound is loose; empirical Rademacher is
  a distribution-dependent tightening over VC.

## Related in this repo

- `vc-dimension` — combinatorial complexity measure.
- `efron-stein-inequality` — variance-of-a-function-of-iid bound
  used to make Rademacher concentrate.
- `information-bottleneck`, `pac-bayes` (future) — companion
  generalisation frameworks.

## Run

```
python techniques/rademacher-complexity/python/rademacher_complexity.py
Rscript techniques/rademacher-complexity/r/rademacher_complexity.R
```

**Refs:** Bartlett, P.L. & Mendelson, S. "Rademacher and Gaussian complexities: risk bounds and structural results." *JMLR*, 3: 463-482, 2002; Mohri, M., Rostamizadeh, A. & Talwalkar, A. *Foundations of Machine Learning*, 2nd ed., MIT Press, 2018.

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
