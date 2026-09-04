# Heterogeneous Treatment Effect + Uplift (Reference §44.7)

Athey & Imbens (2016), Künzel et al. (2019). Estimate the
**conditional average treatment effect** (CATE):

```
τ(x) = E[Y(1) − Y(0) | X = x]
```

## Meta-learners

- **T-learner** — fit `μ_0(x)` and `μ_1(x)` separately on control
  and treated subsets → `τ̂(x) = μ_1 − μ_0`.
- **S-learner** — fit a single model on `(X, T)` → `τ̂(x) =
  μ(x, 1) − μ(x, 0)`.
- **X-learner** — combines T-learner residuals with propensity
  weighting; robust to class imbalance.
- **Causal forest** (Athey-Wager) — honest random forest with
  causal-splitting criterion.

## Uplift evaluation

Rank users by predicted CATE and compare to random ranking via the
**Qini curve** and **Qini score** (area under the uplift curve).

## When to use

- **Targeted marketing** — treat only users predicted to benefit.
- **Precision medicine** — subgroup-specific treatment decisions.
- **Feature rollout** — identify who benefits vs who is harmed.

## When NOT to use

- **Constant treatment effect** — HTE adds noise where there is
  no signal.
- **Small samples** — CATE estimation is high-variance; simple ATE
  is honest.

## Files

- `python/hte_uplift.py` — T-learner + S-learner with GBM base +
  Qini score. Demo (n=3000, p=5, true τ = X₀): CATE correlations
  with truth **T=0.98, S=0.99**; Qini T=0.20, S=0.20, random 0.05
  — 4× lift over random targeting.
- `r/hte_uplift.R` — `grf::causal_forest`, `uplift`,
  `causalToolbox` (R); `causalml`, `econml`, custom (Python).

## Assumptions & caveats

- **Unconfoundedness** — CATE inherits identification from
  randomisation or from correct covariate adjustment.
- **Overlap / positivity** — every X must have both treated and
  untreated observations.
- **Cross-fitting** — honest sample splits or DML de-biasing to
  avoid overfitting on the CATE.
- **Uplift vs response modelling** — uplift targets those who
  *change* behaviour; response modelling targets those who
  *convert* regardless of treatment.

## Related in this repo

- `causal-forest`, `double-ml`, `dr-learner` — advanced HTE
  estimators.
- `ab-test-fundamentals` — the ATE baseline.
- `personalization` (if present) — deployment cousin.

## Run

```
python techniques/hte-uplift/python/hte_uplift.py
Rscript techniques/hte-uplift/r/hte_uplift.R
```

**Refs:** Athey, S. & Imbens, G.W. "Recursive partitioning for heterogeneous causal effects." *PNAS*, 2016; Künzel, S.R., Sekhon, J.S., Bickel, P.J., & Yu, B. "Metalearners for estimating heterogeneous treatment effects using machine learning." *PNAS*, 2019.

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
