# Model-X Knockoffs (Reference §32.5)

Candès, Fan, Janson & Lv (2018) — **FDR-controlled variable selection**
in high-dim regression via "knockoff" copies of the design.

## Algorithm

1. Construct **knockoffs** `X̃` — copies with the same joint law as `X`
   but conditionally independent of `Y` given `X`. For Gaussian
   `X ~ N(0, Σ)`, equi-correlated knockoffs:

   ```
   X̃ = X (I − Σ⁻¹ diag(s))  +  Chol(2 diag(s) − diag(s) Σ⁻¹ diag(s)) Z
       s_j = min(1, 2 λ_min(Σ))
   ```

2. Fit a **statistic** `Z_j` per feature (LASSO absolute-coef, RF
   importance, boosting importance).

3. `W_j = Z_j(original) − Z_j(knockoff)`.

4. **Barber-Candès filter**: pick threshold `τ` — the smallest such
   that `(1 + #{W ≤ −τ}) / #{W ≥ τ} ≤ q`. Select `{j : W_j ≥ τ}`.

## Guarantees

Under exchangeability of knockoffs the selection controls the
**modified FDR** at level `q` in finite sample.

## When to use

- **High-dim regression** where you need a FDR guarantee on selection.
- **Genomics, imaging, finance** — the paper's motivating domains.
- **Any model class** for `Z_j`: LASSO, RF, XGBoost, deep nets.

## When NOT to use

- **Fewer than `1/q` true signals** — the filter cannot select any
  feature at the target level (the demo uses 12 signals for q = 0.15).
- **Very high correlations** — knockoff quality degrades; joint law
  is hard to mimic.
- **Non-Gaussian `X` with unknown law** — Model-X requires the joint
  distribution of `X`; second-order knockoffs work for arbitrary `X`
  under Gaussian assumption on inner products.

## Files

- `python/model_x_knockoffs.py` — from-scratch equi-correlated
  Gaussian knockoffs + LASSO importance + Barber-Candès filter. Demo
  50 trials `n=300, p=40`, 12 signals: **empirical FDR = 0.130
  (target 0.15); power = 1.000**.
- `r/model_x_knockoffs.R` — `knockoff` (R reference); `knockpy`
  (Python).

## Assumptions & caveats

- **Signal count `≥ 1/q`** — the filter needs at least one selection
  to give a valid FDR estimate.
- **Knockoff quality** — smaller `s_j` = tighter mimicry = weaker
  power; larger `s_j` = looser mimicry = better power but risks
  breaking exchangeability. Equi-correlated is the simplest choice;
  SDP-optimised `s` improves power (Candès 2018).
- **Multiple knockoff draws** — average selection stability across
  draws (multi-KO).
- **Importance-statistic choice** — LASSO for linear, RF / XGB for
  nonlinear.
- **Second-order knockoffs** — cover non-Gaussian `X` under weaker
  assumptions.

## Related in this repo

- `debiased-lasso` — CIs, an alternative form of inference.
- `stability-selection` — resampling-based FDR proxy.
- `ridge-lasso-elasticnet`, `adaptive-lasso`, `scad-mcp-penalties` —
  the sparse-regression toolbox.
- `false-discovery-rate-control` (if present) — the classical FDR
  machinery (Benjamini-Hochberg).

## Run

```
python techniques/model-x-knockoffs/python/model_x_knockoffs.py
Rscript techniques/model-x-knockoffs/r/model_x_knockoffs.R
```

**Refs:** Candès, E.J., Fan, Y., Janson, L. & Lv, J. "Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection." *JRSS-B*, 2018; Barber, R.F. & Candès, E.J. "Controlling the false discovery rate via knockoffs." *Annals of Statistics*, 2015.

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
