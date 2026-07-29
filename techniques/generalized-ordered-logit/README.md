# Generalized Ordered Logit / Partial Proportional Odds (Reference §8.35)

Ordinal outcome `Y ∈ {1, ..., J}`. The standard proportional-odds (PO) logit fits

```
logit P(Y ≤ j | x) = αⱼ − xᵀβ          (SAME β for all j)
```

but the PO assumption often fails. Two relaxations:

- **Fully generalized ordered logit** (Peterson & Harrell 1990; Williams 2006): allow each cutpoint its own coefficient vector `βⱼ`.
- **Partial PO**: some covariates share `β`, others get `βⱼ`. Stata's `gologit2` picks this up automatically from a chi-square test at each covariate.

```
logit P(Y ≤ j | x) = αⱼ − xᵀβⱼ
```

Fitting the fully generalized version is equivalent to `J − 1` **independent** binary cumulative logits on `y_j = I(Y ≤ j)`.

## Brant test (Brant 1990)

Wald test that `β₁ = β₂ = ... = β_{J−1}` for each covariate. Rejection = PO fails for that covariate; use partial or fully generalized PO for it.

## Files

- `python/generalized_ordered_logit.py` — fully generalized fit via `J − 1` BFGS binary logits, with a Brant-style Wald test per covariate. Demo recovers `x₁` non-proportional (p = 0.0006) vs `x₂` proportional (p = 0.65).
- `r/generalized_ordered_logit.R` — `VGAM::vglm(cumulative(parallel = FALSE))` + `brant::brant` on a proportional-odds fit.

## When to use

- Any ordinal outcome where the "same effect at every cutpoint" story is suspect (self-rated health, Likert scales, disease-stage progression).
- After a significant Brant test on a PO logit.
- When a specific covariate's effect is expected to differ across categories (e.g. treatment helps mild-to-moderate but not severe cases).

## Assumptions

- Cumulative probabilities remain monotone in `j`; the fully generalized model does **not** enforce this — fitted probabilities can go negative or non-monotone at extreme `x`. Partial PO restores monotonicity when the PO covariates dominate.
- Adequate sample per category for `J − 1` binary logits.

## Run

```
python techniques/generalized-ordered-logit/python/generalized_ordered_logit.py
Rscript techniques/generalized-ordered-logit/r/generalized_ordered_logit.R
```

**Refs:** Peterson, B. & Harrell, F.E. "Partial proportional odds models for ordinal response variables." *Appl. Stat.* 39(2), 205–217, 1990; Brant, R. "Assessing proportionality in the proportional odds model for ordinal logistic regression." *Biometrics* 46(4), 1171–1178, 1990; Williams, R. "Generalized ordered logit / partial proportional odds models for ordinal dependent variables." *Stata J.* 6(1), 58–82, 2006.

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
