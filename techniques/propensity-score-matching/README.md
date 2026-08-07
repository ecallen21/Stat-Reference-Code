# Propensity Score Matching (Reference §15.6)

Observational data: treatment `T` is confounded by observed covariates `X`. Estimate the **average treatment effect on the treated (ATT)**:

```
ATT = E[Y(1) − Y(0) | T = 1]
```

**Propensity score** (Rosenbaum-Rubin 1983):

```
e(x) = Pr(T = 1 | X = x)
```

If **strong ignorability** holds — `(Y(0), Y(1)) ⊥ T | X` — then it also holds when conditioning on `e(X)` instead of the full `X`. So we can match on the 1-D propensity instead of the multivariate `X`.

## 1:1 nearest-neighbor matching

```
1. Fit logistic  T ~ X  → ê(x).
2. For each treated i, find control j* minimizing |ê(x_i) − ê(x_j)|.
3. ATT̂ = mean(Y_i − Y_{j*(i)}).
```

## Balance diagnostics

Report **standardized mean difference** (SMD) per covariate before vs after matching:

```
SMD_k = (mean_treated − mean_control) / √((var_treated + var_control) / 2)
```

Rule of thumb: `|SMD| < 0.1` → good balance.

## Files

- `python/propensity_score_matching.py` — logistic-regression propensity + 1:1 nearest-neighbor matching with replacement + SMD diagnostics. Demo (n = 800, true treatment effect 2.0, confounded assignment): naive difference in means 2.29 (biased); matched ATT 2.17; matched SMDs on x1 and x2 both < 0.05.
- `r/propensity_score_matching.R` — `MatchIt::matchit(T ~ x1 + x2, method = "nearest", replace = TRUE)`.

## When to use

- **Observational treatment effects** where randomization isn't feasible.
- **Balancing covariates** before regression adjustment ("doubly robust" analysis).
- **Communicating** causal identification: matched analysis is easier to explain than parametric adjustment.

## Extensions

- **1:M matching** — more controls per treated, gains efficiency.
- **Caliper** — reject matches beyond `δ · SD(ê)` distance.
- **Kernel matching** — weighted average of controls close in propensity.
- **Coarsened Exact Matching (CEM)** — coarsen covariates, exact-match on the coarsened values.
- **Optimal matching** — global assignment minimizing total distance.
- **Entropy balancing** — reweight controls to match treated moments without matching.

## When NOT to use

- **Common-support violation** — treated units with no near-propensity control mean no ATT can be estimated for them.
- **Poor propensity model** — logistic regression that misses important predictors leaves residual confounding.
- **Unmeasured confounding** — matching can't fix what isn't in `X`. Report a sensitivity analysis (Rosenbaum bounds, E-value).

## Related methods

- **Inverse Probability Weighting** (`inverse-probability-weighting`) — reweight rather than subset.
- **Doubly-robust / TMLE** (`tmle-doubly-robust`) — combine matching / IPW with outcome regression.
- **Difference-in-Differences** (`diff-in-diff`) — different identification strategy exploiting time.
- **Regression Discontinuity** (`regression-discontinuity`) — exploits a treatment cutoff.

## Assumptions & caveats

- **Strong ignorability** (unconfoundedness) — the key untestable assumption.
- **SUTVA** — no interference, no hidden treatment variations.
- **Common support** — propensity distributions of treated and control overlap.

## Run

```
python techniques/propensity-score-matching/python/propensity_score_matching.py
Rscript techniques/propensity-score-matching/r/propensity_score_matching.R
```

**Refs:** Rosenbaum, P.R. & Rubin, D.B. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 70(1), 41–55, 1983; Ho, D.E., Imai, K., King, G. & Stuart, E.A. "MatchIt: nonparametric preprocessing for parametric causal inference." *J. Stat. Softw.* 42(8), 1–28, 2011; Stuart, E.A. "Matching methods for causal inference: a review and a look forward." *Stat. Sci.* 25(1), 1–21, 2010.

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
