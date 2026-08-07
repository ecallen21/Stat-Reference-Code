# TMLE + Doubly-Robust ATE (Reference §15.11)

Doubly-robust semiparametric estimator of the **average treatment effect** under strong ignorability. Combines an outcome regression with a propensity model to gain double protection against misspecification.

## Setup

- `Y` — outcome
- `A` — binary treatment
- `W` — confounders
- `Q̄(A, W) = E[Y | A, W]` — outcome regression
- `g(W) = Pr(A = 1 | W)` — propensity score

Target: `ATE = E[Q̄(1, W) − Q̄(0, W)]`.

## AIPW / doubly-robust estimator (Robins-Rotnitzky-Zhao 1994)

```
ATE_AIPW = mean(
    Q̄(1, W) − Q̄(0, W)
  + A (Y − Q̄(1, W)) / g(W)
  − (1 − A) (Y − Q̄(0, W)) / (1 − g(W))
)
```

**Doubly robust**: consistent if **either** `Q̄` or `g` is correctly specified.

## TMLE (van der Laan-Rubin 2006)

Adds a **targeting step**: after obtaining initial `Q̄` and `g`, update `Q̄` to solve the efficient influence function equation. Fluctuation submodel (continuous outcome, linear):

```
Q̄^*(A, W) = Q̄(A, W) + ε · H(A, W)
H(A, W)   = A / g(W) − (1 − A) / (1 − g(W))          clever covariate
```

Fit `ε` by (weighted) OLS on residuals. Update `Q̄`; recompute ATE as the plug-in.

Both AIPW and TMLE are doubly robust; TMLE is a **plug-in** estimator that respects the parameter-space boundary (better for bounded outcomes and probabilities).

## Files

- `python/tmle_doubly_robust.py` — logistic PS + OLS outcome regression with a treatment-covariate interaction + AIPW and linear-fluctuation TMLE + IC-based SE. Demo (n = 1000, true ATE = 2.0): AIPW ATE = 1.85, SE 0.084; TMLE ATE = 1.85, SE 0.084; naive difference = 2.72 (biased); ε ≈ 0 confirms initial outcome fit was already well-calibrated.
- `r/tmle_doubly_robust.R` — `tmle::tmle(Y, A, W)` with SuperLearner-based Q̄ and g by default.

## When to use

- **Observational treatment effects** where you want robustness to misspecification of either the outcome model or the propensity model.
- **Semiparametric efficiency** — TMLE achieves the semiparametric bound when both Q̄ and g are consistent.
- **Modern causal-inference pipelines** built on SuperLearner / ML for Q̄ and g.

## Extensions

- **Longitudinal TMLE** (LTMLE) — sequential treatments; `ltmle` package.
- **Continuous / categorical treatments** — LMTP (Diaz-van der Laan 2018), `lmtp` R package.
- **Instrumental TMLE** for endogenous treatment.
- **Cross-fitting** (double machine learning; Chernozhukov et al. 2018) — sample-splitting between nuisance-model estimation and target estimation for asymptotic validity with ML nuisance models.

## Assumptions & caveats

- **Positivity**: `g(W) ∈ (0, 1)` — no deterministic treatment. Clip propensities and check the distribution.
- **Strong ignorability + SUTVA** — same as PSM / IPW.
- **Nuisance models**: ML learners (random forests, GBM, neural nets) via SuperLearner are standard; use cross-fitting to avoid overfitting bias.
- **Influence-function SE** is asymptotic; bootstrap for small samples.

## Run

```
python techniques/tmle-doubly-robust/python/tmle_doubly_robust.py
Rscript techniques/tmle-doubly-robust/r/tmle_doubly_robust.R
```

**Refs:** Robins, J.M., Rotnitzky, A. & Zhao, L.P. "Estimation of regression coefficients when some regressors are not always observed." *JASA* 89(427), 846–866, 1994; van der Laan, M.J. & Rubin, D. "Targeted maximum likelihood learning." *Int. J. Biostat.* 2(1), 2006; van der Laan, M.J. & Rose, S. *Targeted Learning*, Springer, 2011; Chernozhukov, V. et al. "Double/debiased machine learning for treatment and structural parameters." *Econ. J.* 21(1), C1–C68, 2018.

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
