# Post-Selection Inference (Reference §32.8)

**Standard CIs / p-values applied AFTER a data-driven selection step
(LASSO, forward stepwise) are anti-conservative** — the same data
selected + fit the model. Post-selection inference (PoSI) constructs
CIs that **condition on the selection event**.

## Approaches

- **Data splitting** (Cox 1975, Wasserman-Roeder 2009) — halve the
  data; select on one, infer on the other. Simple, valid, uses less
  data.
- **Simultaneous PoSI** (Berk-Brown-Buja 2013) — wide simultaneous
  bands over all sub-models.
- **Conditional PoSI** (Lee-Sun-Sun-Taylor 2016) — exact conditional
  distribution given the LASSO selection event.

## When to use

- **Whenever LASSO / stepwise / any adaptive selection is followed by
  CIs / p-values** on selected variables.
- **Regulated / audit settings** where selection bias must be
  controlled.

## When NOT to use

- **Pure prediction tasks** — CIs aren't required.
- **Very small n** — data splitting halves the sample; conditional
  PoSI has more power but is complex.

## Files

- `python/post_selection_inference.py` — naive OLS-after-LASSO vs
  data-split PoSI. 100 trials, `n=200, d=50`, 3 signals. **Data-split
  coverage 0.946** (nominal 0.95); naive coverage 0.861 — clear
  anti-conservatism of the naive approach.
- `r/post_selection_inference.R` — `selectiveInference`, `hdi`, `PoSI`
  (R); `selectinf` (Python).

## Assumptions & caveats

- **Data-split loses efficiency** — only half the data for each step;
  conditional PoSI (Lee 2016) is fully efficient but harder to derive.
- **Selection stability** — with different splits you may get
  different selected sets.
- **Multiple selections** in a pipeline compound the bias.
- **Nonparametric variants** for post-selection permutation tests exist
  (Fithian-Sun-Taylor 2014).

## Related in this repo

- `debiased-lasso`, `model-x-knockoffs`, `stability-selection`,
  `sure-independence-screening`, `scad-mcp-penalties` — the high-dim
  inference toolbox.
- `bootstrap`, `bca-bootstrap`, `jackknife` — resampling siblings.
- `bayesian-linear-regression`, `bayesian-glms` — Bayesian alternative
  (posterior is coherent under selection if the prior is set before
  seeing data).

## Run

```
python techniques/post-selection-inference/python/post_selection_inference.py
Rscript techniques/post-selection-inference/r/post_selection_inference.R
```

**Refs:** Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. "Valid post-selection inference." *Annals of Statistics*, 2013; Lee, J.D., Sun, D.L., Sun, Y. & Taylor, J.E. "Exact post-selection inference, with application to the LASSO." *Annals of Statistics*, 2016; Wasserman, L. & Roeder, K. "High-dimensional variable selection." *Annals of Statistics*, 2009.

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
