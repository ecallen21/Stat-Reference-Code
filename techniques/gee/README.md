# Generalized Estimating Equations (Reference §12.8; also covers §12.24, §12.31)

**Population-averaged / marginal** model for clustered data:

```
g(μ_{ij})  =  X_{ij}' β
```

Solved by iteratively-reweighted score equations with a **working correlation structure** `R(α)`:

```
U(β)  =  Σ_i X_i' D_i V_i⁻¹ (y_i − μ_i)  =  0
        V_i  =  A_i^{1/2} R(α) A_i^{1/2}      (working)
        A_i  =  diag(var(μ_i))                 (GLM variance function)
        D_i  =  diag(dμ/dη)                    (link derivative)
```

**Beauty of GEE (Liang–Zeger 1986):** even if `R(α)` is misspecified, `β̂` is consistent as long as the mean model is correct. Only the **variance** needs the sandwich correction.

## Robust (sandwich) SE

```
Var_robust(β̂)  =  Bread · Meat · Bread
     Bread  =  (Σ_i X_i' D_i V_i⁻¹ D_i X_i)⁻¹
     Meat   =  Σ_i X_i' D_i V_i⁻¹ (y − μ)(y − μ)' V_i⁻¹ D_i X_i
```

Always report the sandwich SE (never the "naive" model-based SE) for GEE.

## §12.24 / §12.31 — GEE vs GLMM: which effect are you estimating?

| | **GEE (marginal)** | **GLMM (subject-specific)** |
|---|---|---|
| Interpretation | "Average change per unit X across the population" | "Change within a subject per unit X, given random effect" |
| Nonlinear link | The two give **different β** (attenuation of GLMM's β vs GEE's β) | |
| Missing data | MCAR-only (unless WGEE) | MCAR + MAR (uses all data) |
| Policy question | Answer with GEE | Answer with GLMM |
| Mechanism question | Answer with GLMM | Answer with GLMM |

Rule of thumb: if your effect statement is *"switching X changes the population's outcome by β on average"*, use GEE. If it's *"switching X changes an individual's outcome by β"*, use GLMM.

## Working-correlation structures

- **Independence** — simplest; still consistent but least efficient.
- **Exchangeable** — all within-cluster pairs equally correlated.
- **AR(1)** — correlation decays with time lag (natural for longitudinal).
- **Unstructured** — arbitrary per-cluster (only if all clusters have same size).

## Files

- `python/gee.py` — from-scratch IRWLS-GEE with sandwich SE; supports gaussian / binomial / poisson × independence / exchangeable / AR(1). β and robust SE match `statsmodels.genmod.gee.GEE` to 4 dp on the demo.
- `r/gee.R` — thin wrapper around `geepack::geeglm`.

## Assumptions

- **Mean model is correct** (link × linear predictor).
- Clusters are independent of each other.
- MCAR missing data (WGEE for MAR — not shipped).
- Enough clusters (rule of thumb: `≥ 20`) for the sandwich SE to be well-calibrated.

## Run

```
python techniques/gee/python/gee.py
Rscript techniques/gee/r/gee.R
```

**Refs:** Liang, K.-Y. & Zeger, S.L. "Longitudinal data analysis using generalized linear models." *Biometrika* 73(1), 13–22, 1986; Zeger, S.L., Liang, K.-Y. & Albert, P.S. "Models for longitudinal data: a generalized estimating equation approach." *Biometrics* 44(4), 1049–1060, 1988; Hardin, J.W. & Hilbe, J.M. *Generalized Estimating Equations*, 2nd ed., Chapman & Hall/CRC, 2013.

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
