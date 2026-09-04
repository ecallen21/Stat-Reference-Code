# Maximum Entropy (Reference §34.9)

Jaynes (1957). Given moment constraints `𝔼[f_j(X)] = μ_j`, the
distribution **maximising entropy** takes the exponential-family form:

```
p*(x)  =  exp( − λ₀ − Σ_j λ_j f_j(x) )
```

Lagrange multipliers `λ_j` are chosen to match the constraints (dual:
minimise the log-partition function).

## Canonical MaxEnt distributions

| Constraint             | MaxEnt distribution           |
|-----------------------|-------------------------------|
| Support only          | Uniform                        |
| Mean                  | Exponential                    |
| Mean + variance       | Gaussian                       |
| Log-mean + log-var    | Log-normal                     |
| Finite support + mean | Truncated exponential (die)   |

## When to use

- **Prior selection** — MaxEnt gives the "least assumptive"
  distribution.
- **Feature-based text classification** — MaxEnt / logistic regression.
- **Species-distribution modelling** — MaxEnt is the standard method
  (Phillips 2006).
- **Statistical physics** — Boltzmann distribution as MaxEnt over
  energy expectation.

## When NOT to use

- **Sample size** — the empirical mean may be a poor moment estimate;
  regularise.
- **Feature explosion** — high-dim MaxEnt with many features is
  overfitting-prone; add priors.

## Files

- `python/maximum_entropy.py` — Newton-iteration on the dual Lagrangian
  for discrete MaxEnt over `{1..6}` with a target mean. Demo:
  - mean=3.5 → uniform (`λ = 0`, `H = log 6`).
  - mean=4.5 → tilted upward (`λ = -0.37`, `H = 1.61`).
  - mean=5.5 → sharply upward (`λ = -1.09`, `H = 0.95`).
  Confirms Gaussian is MaxEnt over `(mean, var)` — Laplace (same σ)
  has less entropy.
- `r/maximum_entropy.R` — `maxentropy`, `dismo`, `ENiRG` (R);
  `scipy.optimize`, `maxentpy`, `Elapid` (Python).

## Assumptions & caveats

- **Constraint identifiability** — some moment sets have no MaxEnt
  distribution (Jaynes' example: infinite mean).
- **Existence of Lagrange multipliers** — moments must be feasible
  under the support.
- **Newton convergence** — the dual is convex; step size or ridge for
  stability.
- **Regularisation** — Gaussian prior on `λ_j` gives MAP MaxEnt (Chen
  1999).

## Related in this repo

- `shannon-entropy` — the objective being maximised.
- `kl-divergence` — MaxEnt subject to prior = minimise KL to prior.
- `information-criteria` — cross-connections via exponential family.
- `logistic-regression`, `multinomial-logit` — special-case MaxEnt
  classifiers.

## Run

```
python techniques/maximum-entropy/python/maximum_entropy.py
Rscript techniques/maximum-entropy/r/maximum_entropy.R
```

**Refs:** Jaynes, E.T. "Information theory and statistical mechanics." *Physical Review*, 1957; Cover, T.M. & Thomas, J.A. *Elements of Information Theory*, Wiley, 2006 (Ch. 12); Phillips, S.J., Anderson, R.P. & Schapire, R.E. "Maximum entropy modeling of species geographic distributions." *Ecological Modelling*, 2006.

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
