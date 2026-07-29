# Linear Mixed Models via REML (Reference §12.2; also covers §12.13, §12.16, §12.20, §12.25, §12.26, §12.27, §12.29, §12.30, §12.32, §12.33)

The workhorse model for **clustered / longitudinal** data:

```
y_{ij}  =  X_{ij}' β  +  Z_{ij}' u_i  +  ε_{ij}

u_i  ~  N(0, G)                (random effects for cluster i)
ε_{ij} ~ N(0, σ²)              (residual)
```

`i` indexes clusters (subjects, schools, sites). `j` indexes observations within a cluster. Standard estimation uses **REML** (Restricted Maximum Likelihood): profile out `β`, then maximize the profile log-likelihood on the covariance parameters.

## BLUPs (Best Linear Unbiased Predictors)

```
û_i  =  G · Z_i' · V_i⁻¹ · (y_i − X_i β̂)         V_i = Z_i G Z_i' + σ² I
```

BLUPs *shrink* raw cluster deviations toward zero — clusters with less data get pulled harder toward the grand mean (**§12.27 partial pooling**).

## ICC (§12.20)

For a random-intercept model:
```
ICC  =  σ²_u / (σ²_u + σ²)
```
Proportion of total variance attributable to the cluster level. Also equals the correlation between two observations in the same cluster.

## Guide sections folded in

| Ref § | What it covers | How this file addresses it |
|---|---|---|
| **§12.13** Three-level / cross-classified | Nested hierarchies (student in class in school) | Extend `Z` and `cluster_ids` to multiple grouping columns |
| **§12.16** Covariance-structure selection | Which random effects to include | Compare nested fits by LR test or AIC |
| **§12.20** ICC from LMM | Cluster-level share of variance | Returned as `ICC_random_intercept` |
| **§12.25** BLUPs / singular fits | Cluster-specific predictions | `blups_head` in the return; singular fits (variance = 0) warned about |
| **§12.26** Crossed random effects | Non-nested groups (rater × item) | Two separate indicator columns in `Z` |
| **§12.27** Shrinkage / partial pooling | Why BLUPs pull toward the mean | Formula above; more data → less shrinkage |
| **§12.29** Choosing RE structure | Maximal (Barr) vs. parsimonious (Bates) | Start maximal; simplify to remove singular components |
| **§12.30** When to use | Clustering / repeats present | Any time observations aren't IID |
| **§12.32** Centering | Grand-mean vs. group-mean | Grand-mean for **between**-cluster effect; group-mean for **within**-cluster |
| **§12.33** Correlation structures | AR(1), CS, unstructured | This code assumes IID residuals; extend via a within-cluster covariance model |

## Files

- `python/linear_mixed_models.py` — from-scratch REML via BFGS on the profile likelihood; random-intercept (and slope-extension) support; BLUPs; ICC. β and variance components match `statsmodels.MixedLM` to 6+ dp.
- `r/linear_mixed_models.R` — thin wrapper around `lme4::lmer` (authoritative), with `nlme::lme` fallback.

## Assumptions

- **Normal random effects** and residuals. Robust to modest departures; more robust to non-normal residuals than to non-normal random effects.
- **Correct random-effect structure**. If ignored variance components are large, fixed-effect SEs are wrong.
- **Balanced or ignorable-missing** data. LMM handles MCAR / MAR out of the box; MNAR needs sensitivity analysis.

## Run

```
python techniques/linear-mixed-models/python/linear_mixed_models.py
Rscript techniques/linear-mixed-models/r/linear_mixed_models.R
```

**Refs:** Laird, N.M. & Ware, J.H. "Random-effects models for longitudinal data." *Biometrics* 38(4), 963–974, 1982; Pinheiro, J.C. & Bates, D.M. *Mixed-Effects Models in S and S-PLUS*, Springer, 2000; Bates, D., Mächler, M., Bolker, B. & Walker, S. "Fitting linear mixed-effects models using lme4." *J. Stat. Soft.* 67(1), 1–48, 2015; Barr, D.J., Levy, R., Scheepers, C. & Tily, H.J. "Random effects structure for confirmatory hypothesis testing: keep it maximal." *J. Mem. Lang.* 68(3), 255–278, 2013; Bates, D., Kliegl, R., Vasishth, S. & Baayen, H. "Parsimonious mixed models." *arXiv:1506.04967*, 2015.

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
