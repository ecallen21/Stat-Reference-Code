# Mixed Logit (Random-Coefficient MNL) (Reference §47.54)

McFadden & Train (2000). Extends multinomial logit by letting the
taste coefficients be **random** across individuals:

    β_n ~ f(β | θ),   P_n(i) = ∫ softmax(x_ni′ β)_i · f(β | θ) dβ.

Integral approximated by simulated log-likelihood over Halton or
QMC draws. Removes IIA (independence of irrelevant alternatives)
and allows correlated errors across alternatives.

## Files

- `python/mixed_logit_mnl.py` — simulated MLE with Halton draws
  and independent-normal taste distribution. Demo (n=800, J=3
  alternatives, K=2 attributes, μ=(1, −0.5), σ=(0.4, 0.3)):
  MLE μ̂=(0.96, −0.49), σ̂=(0.23, 0.33) with R=200 Halton draws.
- `r/mixed_logit_mnl.R` — `mlogit`, `gmnl`, `apollo` (R);
  `xlogit`, `pylogit`, from-scratch (Python).

## When to use

- **Transport choice / travel modes** — original McFadden domain.
- **Marketing conjoint** with heterogeneous preferences.
- **Health-utility discrete choice** (patient preferences).
- **Any choice model where IIA is suspect** — mixed logit
  captures cross-alternative correlations.

## When NOT to use

- **Very small samples** — random coefficients need enough
  variation.
- **Fully specified nesting** — nested-logit or GEV is more
  parsimonious.
- **Panel with unobserved heterogeneity in scale** — use scale-
  heterogeneity mixed logit (S-MNL, G-MNL).

## Assumptions & caveats

- **Distribution choice** — normal, log-normal (for sign
  constraints), triangular; results sensitive to specification.
- **Simulation draws** — 100–500 Halton draws typical; more for
  high-dim β.
- **Local optima** — non-convex likelihood; use multiple starts.
- **Identifiability of σ** — needs enough within-choice variation
  in x.

## Related in this repo

- `multinomial-logistic`, `conjoint-choice`,
  `plackett-luce-ranking`, `bradley-terry` — discrete-choice
  cousins.
- `bayesian-hierarchical-models`, `bayesian-glms` — Bayesian
  random-coefficient alternatives.
- `latin-hypercube-sampling`, `quasi-monte-carlo-sobol` — QMC
  integration tools.
- `iv-2sls`, `oaxaca-blinder` — econometric complements.

## Run

```
python techniques/mixed-logit-mnl/python/mixed_logit_mnl.py
Rscript techniques/mixed-logit-mnl/r/mixed_logit_mnl.R
```

**Refs:** McFadden, D. & Train, K. "Mixed MNL models for discrete response." *J Appl Econometrics* 15(5): 447-470, 2000; Train, K. *Discrete Choice Methods with Simulation.* Cambridge Univ Press, 2nd ed., 2009.

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
