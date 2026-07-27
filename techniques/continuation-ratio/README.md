# Continuation-Ratio Model for Ordinal Outcomes (Reference §8.9)

For an ordinal outcome `Y ∈ {1, ..., K}`, the continuation-ratio (CR) model decomposes the ordered response into `K − 1` **sequential** binary transitions:

```
logit P(Y > k | Y ≥ k, X)  =  α_k + X·β_k       for k = 1, ..., K−1
```

Interpretation: *"Given you're still at stage k, what's the odds of advancing to k+1 or beyond?"* Natural for **irreversible stage-progression** processes: cancer stage, educational attainment, disease severity, retirement transitions.

## Two variants

| Variant | Model | Fit as |
|---|---|---|
| Category-specific | Each `β_k` can differ (full flexibility) | `K − 1` **separate** binary logistic regressions on the subset with `Y ≥ k` |
| Proportional (common β) | Single `β` shared across all transitions | ONE binary logistic on **stacked** data with `α_k` intercept dummies |

Category-specific is always the more flexible fit. Test whether the simpler proportional model suffices with a **likelihood-ratio test**:

```
LR = 2 (ll_category − ll_common)  ~  χ² with (K − 1)·(p − 1) df
```

Large p ⇒ proportional model is fine (parsimony). Small p ⇒ transitions genuinely differ; use category-specific.

## Why CR (vs. proportional-odds / cumulative logit)?

- CR fits the **conditional** transition dynamics — natural when categories are stages you can only reach by passing through the previous ones. PO models the **cumulative** cutoffs — natural when categories are thresholds on an underlying continuous scale.
- CR's binary sub-models are **independent** given the marginal counts at each stage, so implementation is trivially K−1 binomial GLMs. Standard software (`vglm(sratio=...)`, `rms::orm`) also handles it.

## Files

- `python/continuation_ratio.py` — both variants (category-specific + common β) fit via IRLS-based logistic; LR test comparing them.
- `r/continuation_ratio.R` — both variants via `stats::glm.fit`; cross-checks `VGAM::vglm(sratio=..., parallel=TRUE)` for the common-β variant.

## Assumptions

- `Y` is **ordered** and the ordering is meaningful (transition k → k+1 is the process of interest).
- Given the covariates, transition attempts at successive stages are **conditionally independent** — reasonable when reaching stage k doesn't itself modify the transition mechanism beyond what's in `X`.

## Run

```
python techniques/continuation-ratio/python/continuation_ratio.py
Rscript techniques/continuation-ratio/r/continuation_ratio.R
```

**Refs:** Fienberg, S.E. *The Analysis of Cross-Classified Categorical Data*, 2nd ed., Springer, 2007 (Ch. 6); Agresti, A. *Categorical Data Analysis*, 3rd ed., Wiley, 2013 (Ch. 8.4); Cox, C. "Location–scale cumulative odds models for ordinal data." *J. R. Stat. Soc. C* 44(3), 349–361, 1995.

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
