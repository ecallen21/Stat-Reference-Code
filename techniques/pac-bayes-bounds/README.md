# PAC-Bayes Bounds (Reference §46.15)

McAllester (1999); Catoni (2007). Generalisation bounds for
**randomised (posterior) hypotheses** `Q`, controlled by the
KL divergence from a data-independent prior `P`.

## McAllester bound

With probability ≥ 1 − δ over the iid sample:

    E_{h ~ Q} L(h)  ≤  E_{h ~ Q} L̂_n(h)
                      + √( (KL(Q ‖ P) + log(2√n / δ)) / (2n) )

## Tighter variants

- **Catoni (2007)** — refined constants for bounded losses.
- **Seeger (2002)** — binary KL inversion (best-in-class for
  classification).
- **Maurer (2004)** — variance-adaptive bounds via Bernstein.

## Files

- `python/pac_bayes_bounds.py` — closed-form McAllester + simplified
  Catoni bounds with Gaussian prior / posterior. Demo (n=500, δ=0.05):
  bound stays tight (0.09 → 0.18) as long as `Q` is close to `P`
  (KL 1.13 → 1.21); shifted posteriors (μ_q = 1.0, KL = 3.13) pay a
  big penalty (bound = 0.59).
- `r/pac_bayes_bounds.R` — from-scratch; no CRAN package.

## When to use

- **Theoretical generalisation for randomised classifiers /
  ensembles** — SVMs, Bayesian NNs, dropout networks.
- **Certified deep-learning bounds** — best-in-class non-vacuous
  bounds for MNIST / CIFAR are PAC-Bayes-based (Dziugaite &
  Roy 2017).
- **Prior-guided model selection** — pick `Q` to trade empirical
  loss against `KL(Q ‖ P)`.

## When NOT to use

- **Deterministic hypotheses** — PAC-Bayes requires a distribution
  over hypotheses; use VC / Rademacher for deterministic.
- **Unbounded losses** — needs a bounded / sub-Gaussian loss;
  clip or use variance-adaptive versions.
- **Vacuous bound** — with priors chosen badly, the bound can be
  > 1; use a data-dependent prior + differential-privacy trick.

## Assumptions & caveats

- **Prior `P` must be data-independent** — pre-registered before
  seeing data (or use PAC-Bayes with differentially private
  prior).
- **Bounded loss** — usually loss ∈ [0, 1].
- **iid sample** — martingale extensions exist for dependent data.
- **KL divergence** — often analytically intractable; use closed
  Gaussian KL or Monte-Carlo approximation.
- **Optimising the bound directly** yields SGD-style objectives
  (PAC-Bayes training).

## Related in this repo

- `rademacher-complexity`, `vc-dimension`, `efron-stein-inequality`
  — sibling generalisation frameworks.
- `bayesian-neural-network`, `deep-ensembles`, `mc-dropout` — the
  randomised hypotheses PAC-Bayes evaluates.
- `information-bottleneck` — an information-theoretic cousin.

## Run

```
python techniques/pac-bayes-bounds/python/pac_bayes_bounds.py
Rscript techniques/pac-bayes-bounds/r/pac_bayes_bounds.R
```

**Refs:** McAllester, D. "PAC-Bayesian model averaging." *COLT*, 1999; Catoni, O. "PAC-Bayesian supervised classification." *IMS Monograph*, 56, 2007; Seeger, M. "PAC-Bayesian generalisation error bounds for Gaussian process classification." *JMLR*, 3: 233-269, 2002.

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
