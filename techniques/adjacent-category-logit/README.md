# Adjacent-Category Logit for Ordinal Outcomes (Reference §8.10)

For an ordinal outcome `Y ∈ {1, ..., K}`, the adjacent-category (AC) logit model contrasts each level directly with the next level up:

```
log P(Y = k) / P(Y = k+1)  =  α_k − β · X       for k = 1, ..., K−1
```

## AC vs. other ordinal models

| Model | Contrast |
|---|---|
| Cumulative logit (proportional odds) | `above k` vs. `at or below k` |
| Continuation ratio | `above k` vs. `at k, given ≥ k` |
| **Adjacent-category** | **`at k` vs. `at k+1`** (direct pairwise) |

AC is mathematically equivalent to a **multinomial logit with a linear-scaling constraint** on the coefficient vector across categories. That constraint keeps AC a genuine *ordinal* model while making implementation a straightforward multinomial fit.

## Common-β form (fit here)

Referenced to the top category `K`:

```
log P(Y = k) / P(Y = K)  =  γ_k − (K − k) · X · β
```

`β` shows up multiplied by `(K − k)` — the "distance" from the baseline. **Sign convention**: positive `β` means positive `X` shifts probability toward higher categories (the natural direction).

## Files

- `python/adjacent_category_logit.py` — common-β fit via BFGS on the multinomial log-likelihood; category-specific fit via K−1 independent binary logistics on adjacent pairs.
- `r/adjacent_category_logit.R` — same, plus `VGAM::vglm(acat=..., parallel=TRUE, reverse=TRUE)` as library cross-check.

## Assumptions

- `Y` is ordered.
- **Common-β form**: assumes the ordinal shift is proportional across categories. Test by comparing to the pairwise fit (LR test) if in doubt.
- Standard multinomial-logit assumptions on the linear predictor.

## Run

```
python techniques/adjacent-category-logit/python/adjacent_category_logit.py
Rscript techniques/adjacent-category-logit/r/adjacent_category_logit.R
```

**Refs:** Agresti, A. *Categorical Data Analysis*, 3rd ed., Wiley, 2013 (Ch. 8); Simonoff, J.S. *Analyzing Categorical Data*, Springer, 2003; Fienberg, S.E. *The Analysis of Cross-Classified Categorical Data*, 2nd ed., Springer, 2007.

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
