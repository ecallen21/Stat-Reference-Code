# Basket Trial Design (Reference §47.264)

Berry et al (2013); Simon et al (2016). A SINGLE drug tested
across `K` different disease cohorts (tumour types sharing a
biomarker). Independent per-basket analyses waste power;
Bayesian HIERARCHICAL BORROWING pools information via a
shared random-effect prior:

```
y_k ~ Binomial(n_k, p_k)
logit(p_k) = θ_k
θ_k ~ Normal(μ, τ²)              borrow across baskets
μ ~ Normal(0, 10²);  τ ~ HalfNormal(1)
```

Baskets with similar effects "borrow strength" (shrink toward
each other); one clearly-null basket is protected by τ
learning small values.

## Files

- `python/basket_trial_design.py` — Random-walk MH sampler
  for the hierarchical logistic model. Demo: 5 baskets, 20
  patients each. Truth: 4 responder baskets at p≈0.30 and 1
  null basket at p=0.05. Posterior mean shrinks responder
  baskets slightly toward each other while the null basket
  stays at ≈0.14 (independent p̂=0.10 was slightly pulled up
  by borrowing, still separated from responders). Posterior
  τ mean 0.71 reflects moderate heterogeneity.
- `r/basket_trial_design.R` — `bhmbasket`, `basket`,
  `rstanarm::stan_glmer`, `brms` (R); PyMC, from-scratch MH
  (Python).

## When to use

- **Biomarker-driven oncology** — one drug across many tumour
  types sharing a molecular signature.
- **Rare-disease baskets** — small `n_k` per basket makes
  borrowing valuable.
- **Interim decisions per basket** — the Bayesian framework
  gives per-basket posterior probability of activity.

## When NOT to use

- **Very heterogeneous baskets** where borrowing is
  inappropriate — use independent per-basket priors (or
  EXNEX-style mixture).
- **Regulatory-approval endpoint per basket** — some agencies
  require independent per-basket p-values; borrowing may not
  count.

## Assumptions & caveats

- **Exchangeability assumption** — the Bayesian model treats
  baskets as exchangeable draws from N(μ, τ²); check with
  EXNEX (part-exchangeable, part-nonexchangeable) for
  robustness.
- **Prior on τ** — half-normal(1) is default; sensitivity to
  a half-Cauchy(1) is worth reporting.
- **Type-I inflation** at the per-basket level — Bayesian
  activity thresholds must be calibrated to preserve
  false-positive rates per basket.
- **Interim analyses** typically use posterior probability
  thresholds (e.g., P(p_k > 0.20 | data) > 0.85).

## Related in this repo

- `bayesian-hierarchical-models` — the general framework.
- `platform-trial-design` — multi-arm cousin.
- `bayesian-ab-testing` — small-`K` analogue.

## Run

```
python techniques/basket-trial-design/python/basket_trial_design.py
Rscript techniques/basket-trial-design/r/basket_trial_design.R
```

**Refs:** Berry, S.M. et al. "Bayesian hierarchical modeling of patient subpopulations: Efficient designs of phase II oncology clinical trials." *Clinical Trials*, 10(5): 720-734, 2013; Simon, R. et al. "The Bayesian basket design for genomic variant-driven phase II trials." *Semin. Oncol.*, 43(1): 13-18, 2016.

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
