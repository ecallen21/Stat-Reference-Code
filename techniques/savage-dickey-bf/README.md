# Savage-Dickey Density-Ratio Bayes Factor (Reference §47.59)

Dickey (1971); Wagenmakers, Lodewyckx, Kuriyal & Grasman (2010). For
NESTED models `H₀: θ = θ₀` vs `H₁: θ` free, the Bayes factor
simplifies to a density ratio at the null value:

    BF_{01}  =  p(θ₀ | y, H₁) / p(θ₀ | H₁).

Only requires **posterior + prior evaluated at θ₀** — no marginal-
likelihood integral, no bridge sampling. Works whenever you can
sample or evaluate the posterior at θ₀.

## Files

- `python/savage_dickey_bf.py` — closed-form Gaussian
  conjugate example. Demo (n=50, prior μ ~ N(0, 4), σ²=1):
  - true μ=0.0 → BF₀₁ = 9.4 (moderate for H₀)
  - true μ=0.2 → BF₀₁ = 3.7
  - true μ=0.5 → BF₁₀ = 70.4 (strong for H₁)
  - true μ=1.0 → BF₁₀ = 1.2 × 10⁷ (decisive).
- `r/savage_dickey_bf.R` — `BayesFactor`, `bridgesampling`,
  `polspline` (R); `PyMC`, `arviz`, from-scratch (Python).

## When to use

- **Nested Bayesian model comparison** — test H₀: θ = θ₀.
- **Psychology, cognitive-science hypothesis tests** — replaces
  frequentist NHST with graded evidence.
- **When bridge sampling is overkill** — Savage-Dickey trivial
  for nested cases with tractable posterior.
- **Sensitivity to prior** — easy to redo under multiple priors.

## When NOT to use

- **Non-nested models** — Savage-Dickey doesn't apply; use bridge
  sampling, harmonic mean, or WAIC / LOO comparison.
- **Improper prior** on θ — BF undefined; use proper priors.
- **Complex posterior with no density estimator** — use KDE, but
  bias-in-the-tail is a real risk near θ₀.

## Assumptions & caveats

- **Point-null H₀** — only for exact-equality nulls; use
  region-of-practical-equivalence (ROPE) for interval nulls.
- **Prior sensitivity** — BF is not scale-free in the prior;
  report robustness across choices (Jeffreys, Cauchy, informed).
- **Posterior density at θ₀** — Monte-Carlo or KDE; error bars via
  bootstrap.
- **Jeffreys 1961 scale** — BF > 3 moderate, > 10 strong, > 100
  decisive; treat as ordinal.

## Related in this repo

- `bridge-sampling-evidence` — general marginal-likelihood
  estimator when Savage-Dickey doesn't apply.
- `bayesian-model-comparison`, `bayesian-model-averaging`,
  `posterior-predictive-checks` — Bayesian model workflow.
- `bayesian-ab-testing` — Bayes-factor A/B testing analog.
- `credible-intervals-hpd`, `variational-inference`,
  `laplace-approximation` — posterior tooling.

## Run

```
python techniques/savage-dickey-bf/python/savage_dickey_bf.py
Rscript techniques/savage-dickey-bf/r/savage_dickey_bf.R
```

**Refs:** Dickey, J.M. "The weighted likelihood ratio, linear hypotheses on normal location parameters." *Ann Math Stat* 42: 204-223, 1971; Wagenmakers, E.-J., Lodewyckx, T., Kuriyal, H. & Grasman, R. "Bayesian hypothesis testing for psychologists: A tutorial on the Savage-Dickey method." *Cognitive Psychology* 60(3): 158-189, 2010.

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
