# Hausman Test (Reference §35.2)

Hausman (1978). Test whether the **random-effects (RE)** estimator is
consistent, or whether the more robust **fixed-effects (FE)** should
be used.

## Hypotheses

- H₀: `𝔼[α_i | x_it] = 0` — RE consistent + efficient.
- H₁: `𝔼[α_i | x_it] ≠ 0` — RE inconsistent; use FE.

## Statistic

```
H  =  (β̂_FE − β̂_RE)ᵀ (V̂_FE − V̂_RE)⁻¹ (β̂_FE − β̂_RE)   ~   χ²(k)   under H₀
```

## Wooldridge auxiliary-regression form (numerically robust)

Augment the RE regression with within-demeaned regressors `x_it − x̄_i`
and jointly test that their coefficients are zero. Equivalent to
Hausman under H₀ and avoids the negative-`dV` singular cases.

## When to use

- **Panel-data model choice** between RE and FE.
- **Diagnostic** for endogeneity of unit effects.
- **Comparison of any two estimators** where one is consistent under
  a stronger assumption (Hausman-Wu, Hausman-Taylor).

## When NOT to use

- **Time-invariant regressors of interest** — FE eliminates them; RE
  needed even if H₀ rejects.
- **Heteroscedastic / clustered errors** — the classical statistic is
  invalid; use auxiliary-regression form with cluster-robust SEs.

## Files

- `python/hausman_test.py` — from-scratch FE via within-demeaning +
  RE via quasi-demeaning + **Wooldridge augmented regression** for a
  numerically stable Hausman. Demo:
  - **Case A** (unit effects independent of x): H = 2.97, p = 0.085 —
    fail to reject; RE OK.
  - **Case B** (unit effects correlated with x): H = 290, p < 0.001 —
    REJECT RE; RE β̂ = 1.86 is severely biased vs FE β̂ = 1.52.
- `r/hausman_test.R` — `plm::phtest` (R reference);
  `linearmodels` + auxiliary regression (Python).

## Assumptions & caveats

- **PSD violation of `dV`** — the classical form can give negative
  numerators in finite samples; the Wooldridge auxiliary regression
  is preferred (used here).
- **Cluster-robust variant** — replace OLS SEs with cluster-robust
  SEs in the auxiliary regression.
- **Time-invariant regressors** are dropped in FE — the test is only
  informative for the time-varying subset.
- **Alternative**: Chamberlain / Mundlak specification adds unit
  means of x as regressors; if their coefficient is 0, RE = FE.

## Related in this repo

- `fixed-effects-panel` — the FE estimator.
- `random-effects` (adjacent) — the RE estimator.
- `arellano-bond-gmm` — dynamic panel where FE alone is inconsistent.
- `sur-regression`, `iv-2sls` — sibling econometric methods.

## Run

```
python techniques/hausman-test/python/hausman_test.py
Rscript techniques/hausman-test/r/hausman_test.R
```

**Refs:** Hausman, J.A. "Specification tests in econometrics." *Econometrica*, 1978; Wooldridge, J.M. *Econometric Analysis of Cross Section and Panel Data*, MIT Press, 2002 (Ch. 10); Baltagi, B.H. *Econometric Analysis of Panel Data*, Wiley, 2005.

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
