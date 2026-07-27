# Restricted Mean Survival Time (Reference §11.29; also covers §11.67)

**RMST(τ)** = expected survival time truncated at horizon τ:

```
RMST(τ)  =  ∫₀^τ  S(u) du
```

Estimated by integrating the KM curve:

```
RMST̂(τ)  =  Σ_{t_j ≤ τ}  Ŝ(t_{j-1}) · (t_j − t_{j-1})  +  Ŝ(t_last) · (τ − t_last)
```

**SE** (Andersen-Hansen-Klein 2004):

```
Var(RMST̂(τ))  =  Σ_{t_j ≤ τ}  [∫_{t_j}^τ Ŝ(u) du]²  ·  d_j / (n_j (n_j − d_j))
```

Between-group **RMST difference test**: `(RMST_A − RMST_B) / √(Var_A + Var_B) ~ N(0, 1)` under `H₀`.

## Why RMST over Hazard Ratio (§11.67)

- **Time-scale interpretation** — "X extends life by 3.2 months, on average, up to 5 years" is directly meaningful.
- **No proportional-hazards assumption** — HR is only well-defined when hazards are PH; RMST always is.
- **Well-defined under crossing / plateauing hazards** — immunotherapy trials often have delayed separation of survival curves; the log-rank test loses power and HR is misleading, but the RMST difference stays valid and interpretable.
- **Robust to censoring pattern** at the horizon — as long as τ is inside where you have adequate follow-up.

## When HR is still preferred

- Truly proportional hazards, all-cause mortality, and you specifically want a rate ratio.
- Model-based inference with covariate adjustment (Cox).
- Regulatory tradition in some settings.

## Files

- `python/rmst.py` — RMST from the KM integral + Andersen SE + between-group difference test. Recovers theoretical RMST for exponential DGP within CI.
- `r/rmst.R` — from-scratch + `survRM2::rmst2` (the standard R implementation).

## Assumptions

- Independent right-censoring.
- Horizon τ should be within the range where both groups have adequate at-risk numbers — a common rule is "not past the last event time in the smaller group" or a pre-specified clinically meaningful time (e.g. 5 years).

## Run

```
python techniques/rmst/python/rmst.py
Rscript techniques/rmst/r/rmst.R
```

**Refs:** Andersen, P.K., Hansen, M.G. & Klein, J.P. "Regression analysis of restricted mean survival time based on pseudo-observations." *Lifetime Data Anal.* 10(4), 335–350, 2004; Uno, H., Claggett, B., Tian, L. *et al.* "Moving beyond the hazard ratio in quantifying the between-group difference in survival analysis." *J. Clin. Oncol.* 32(22), 2380–2385, 2014; Royston, P. & Parmar, M.K.B. "Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome." *BMC Med. Res. Methodol.* 13, 152, 2013.

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
