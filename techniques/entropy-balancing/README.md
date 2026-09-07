# Entropy Balancing (Reference §15.55)

Hainmueller (2012). Directly reweight control units so their
**covariate moments EXACTLY match** treated moments, subject to
keeping weights as close to a base distribution as possible in KL
sense.

## Optimisation

```
min  Σ w_i log(w_i / q_i)
s.t. Σ w_i c_ij = m_j   for each moment j
     Σ w_i     = 1
```

**Dual** solution: `w_i = exp(−Σ λ_j c_ij) / Z(λ)`; solve for `λ`
by Newton on moment-matching residuals.

## When to use

- **Observational causal inference** where you want to guarantee
  exact covariate balance rather than approximate PS balance.
- **When PS estimation is unreliable** (misspecified logistic).

## When NOT to use

- **Very high-dim covariates** — infeasibility if moments can't
  be matched.
- **Extreme covariate separation** — no feasible weighting
  exists; report the failure.

## Files

- `python/entropy_balancing.py` — Newton-based dual solver
  (custom). Demo (n_c=2000, n_t=500, p=3, targets = treated
  first + second moments): treated means (0.50, 0.42, 0.57)
  matched exactly by weighted controls; ESS = 912 of 2000
  (weights concentrate but stay diffuse).
- `r/entropy_balancing.R` — `WeightIt::weightit(method='ebal')`,
  `ebal`, `survey::calibrate` (R); custom + `econml` (Python).

## Assumptions & caveats

- **Feasibility** — targets must lie in the convex hull of control
  moments; otherwise no solution.
- **Moment choice** — 1st (mean) is most common; add 2nd (variance)
  and higher moments as needed.
- **Positivity implicit** — infeasibility flags weak overlap.
- **SEs via bootstrap** — plug-in inference under-estimates.
- **Combine with outcome regression** for double robustness (EB +
  augmented estimator).

## Related in this repo

- `iptw`, `aipw-doubly-robust`, `coarsened-exact-matching` —
  alternative reweighting / matching.
- `stable-balancing-weights` (if present) — related family.

## Run

```
python techniques/entropy-balancing/python/entropy_balancing.py
Rscript techniques/entropy-balancing/r/entropy_balancing.R
```

**Refs:** Hainmueller, J. "Entropy balancing for causal effects: a multivariate reweighting method to produce balanced samples in observational studies." *Political Analysis*, 2012.

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
