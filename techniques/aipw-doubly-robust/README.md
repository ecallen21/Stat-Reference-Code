# AIPW — Augmented IPW / Doubly Robust (Reference §15.8)

Robins-Rotnitzky-Zhao (1994), Kang-Schafer (2007). Combines two
nuisance estimates for **doubly robust** ATE:

- `e(X) = P(T=1|X)` — propensity model.
- `m_t(X) = E[Y | T=t, X]` — outcome regression.

```
ATE_AIPW = E[ m_1(X) − m_0(X)
             + T · (Y − m_1(X)) / e(X)
             − (1 − T) · (Y − m_0(X)) / (1 − e(X)) ]
```

Consistent if **either** e(X) or `m_t(X)` is correctly specified;
efficient under both.

## When to use

- **Observational causal inference** where you don't want to bet
  on a single nuisance model.
- **DML / cross-fitting** setups — AIPW is the canonical building
  block.

## When NOT to use

- **Near-violated positivity** — extreme weights blow variance;
  use overlap weighting or TMLE which targets more variance
  reduction.
- **Both models wildly misspecified** — no method rescues that.

## Files

- `python/aipw_doubly_robust.py` — AIPW + g-formula + IPTW side-
  by-side, with optional 2-fold cross-fitting. Demo (n=2000,
  nonlinear confounding, true ATE=0.4): naive estimates biased;
  AIPW recovers **0.475 without cross-fit, 0.463 with cross-fit**.
- `r/aipw_doubly_robust.R` — `AIPW`, `CausalGAM`, `tmle`,
  `drtmle` (R); `DoubleML::DoubleMLIRM`,
  `econml.dr.DRLearner` (Python).

## Assumptions & caveats

- **Cross-fitting** is required for asymptotic inference when
  using ML nuisance models (Chernozhukov 2018).
- **Weight trimming** to bound `e` away from 0 / 1.
- **Report both** the g-formula estimate and the IPTW estimate
  alongside AIPW for diagnostic transparency.
- **TMLE** further targets the parameter for efficiency; AIPW is a
  one-step alternative.

## Related in this repo

- `iptw`, `tmle-doubly-robust`, `dml-double-ml`, `g-computation`
  — the causal-inference toolbox.
- `propensity-score-matching` — a matching alternative.

## Run

```
python techniques/aipw-doubly-robust/python/aipw_doubly_robust.py
Rscript techniques/aipw-doubly-robust/r/aipw_doubly_robust.R
```

**Refs:** Robins, J.M., Rotnitzky, A., & Zhao, L.P. "Estimation of regression coefficients when some regressors are not always observed." *JASA*, 1994; Kang, J.D.Y. & Schafer, J.L. "Demystifying double robustness." *Statistical Science*, 2007.

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
