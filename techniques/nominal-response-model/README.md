# Bock Nominal Response Model (Reference §22.x extra)

For an item with `K` **unordered** response categories, the probability of
each response is a multinomial-logit function of the latent trait:

```
P(U_ij = k | θ_i) = exp(a_{jk} θ_i + c_{jk}) / Σ_l exp(a_{jl} θ_i + c_{jl})
```

Identifiability: fix `a_{j0} = 0, c_{j0} = 0` for a reference category (or
sum-to-zero constraints).

## Why "nominal"

Multiple-choice items where the **distractors** carry diagnostic information:
choosing distractor B vs distractor C tells you something different about the
respondent's `θ`, and neither one is "more correct" than the other. In the
2PL/3PL family only "right vs wrong" is modelled, throwing that information
away.

## Special cases and relatives

- **PCM / GPCM** (see `partial-credit-model`) — ordered categories; category slopes are constrained to be a monotone function of `θ`.
- **GRM** (see `graded-response-model`) — cumulative logit for ordered categories.
- **Multiple-choice model (Thissen-Steinberg 1984)** — adds a lower-asymptote for guessing at random from a category.
- **Log-linear IRT (Kelderman)** — same family via a Rasch-like log-linear specification.

## When to use

- **Multiple-choice items** — investigating whether distractors are diagnostic.
- **Nominal survey items** — brand choice, preferred candidate, response to an open coding scheme.
- **Diagnostic misconceptions** — students who choose distractor B tend to have a specific misunderstanding.
- **DIF for unordered items** — the Mantel-Haenszel and logistic-DIF approaches don't directly generalise; NRM does.

## Files

- `python/nominal_response_model.py` — Bock NRM by MML with 21-point Gauss-Hermite quadrature; per-item multinomial-logit M-step; EAP `θ` update; explicit identification by rescaling `θ` (compensating `a`, `c`). Demo (n=500, J=8, K=4): cor(a_hat, a) = 0.97 across 24 cells, cor(c_hat, c) = 0.85, cor(θ_hat, θ) = 0.82; example item slopes recovered on the right scale.
- `r/nominal_response_model.R` — `mirt::mirt(itemtype='nominal')`, `mirt::itemplot(type='trace')`, `mirt::fscores(method='EAP')`.

## Assumptions & caveats

- **Reference category** is arbitrary — different choices give different `a, c` but the same probabilities. Fit statistics and predictions are invariant.
- **Identifiability** of `θ` scale — pin down `θ ~ N(0, 1)` or fix one category's `a_{jk} = 1`. Without it MML can drift.
- **Many parameters per item** — `2 (K − 1)` free per item; needs a fairly large sample for stable estimation. Rule of thumb: at least 5–10 responses per category.
- **Non-identifiability of empty categories** — a distractor never chosen has undefined parameters; drop or pool it.
- **Ordered items in disguise** — if your categories are ordered, prefer GRM / GPCM: fewer parameters, respects the order, and interprets more cleanly.

## Run

```
python techniques/nominal-response-model/python/nominal_response_model.py
Rscript techniques/nominal-response-model/r/nominal_response_model.R
```

**Refs:** Bock, R.D. "Estimating item parameters and latent ability when responses are scored in two or more nominal categories." *Psychometrika* 37(1), 29–51, 1972; Thissen, D. & Steinberg, L. "A response model for multiple-choice items." *Psychometrika* 49(4), 501–519, 1984.

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
