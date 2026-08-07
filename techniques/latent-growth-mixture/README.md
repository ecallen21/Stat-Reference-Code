# Latent Growth Mixture Model (Reference §12.13)

Longitudinal data with **unobserved subpopulations**, each following its own growth trajectory:

```
y_ij | class = k  ~ Normal(α_k + β_k · t_ij + b_i, σ_k²)
b_i               ~ Normal(0, τ_k²)          (subject-specific random intercept)
class             ~ Categorical(π_1, ..., π_K)
```

Extends the **group-based trajectory model (GBTM)** of Nagin (1999) — which allows only class-specific fixed trajectories — with subject-level random effects within each class (Muthén 2004).

## Estimation (EM)

- **E-step**: posterior class probability for each subject given all its observations.
- **M-step**: within each class, fit a random-intercept LMM by weighted MLE.

Number of classes `K` chosen by BIC, entropy, and substantive plausibility.

## Files

- `python/latent_growth_mixture.py` — simplified EM for `K` linear trajectories with class-specific `α_k, β_k` and shared residual σ² (fixed-only, GBTM flavor). Demo (N = 300, T = 6, K = 3 with true π = (0.4, 0.4, 0.2)): with 6 random restarts, recovers all three intercepts and slopes to 3 decimals; 100% best-permutation classification accuracy.
- `r/latent_growth_mixture.R` — `lcmm::hlme` (Proust-Lima; full LGMM with random effects and class-varying variances).

## When to use

- **Latent classes of trajectories** — different subgroups of subjects follow qualitatively different trajectories (rapid decline vs stable vs improvement).
- **Personalized-medicine framing** — identify "profiles" that respond differently to treatment.
- **Development / education** — distinct developmental pathways.

## Related methods

- **Group-based trajectory** (GBTM, Nagin 1999) — same idea but without random effects within class.
- **Growth-curve model** — single-class random-effects linear growth.
- **Discrete latent-variable models / SEM growth mixtures** (Mplus, `lavaan`, `blavaan`) — richer class-varying covariance structures.

## Assumptions & caveats

- **Local optima** — EM is notorious for LGMM. Use ≥ 5 random restarts and BIC to pick.
- **Class number `K`**: use BIC, entropy (`E = 1 − Σ_ik γ_ik log γ_ik / (N log K)` — near 1 means well-separated classes), and substantive plausibility. BIC alone often points to more classes than are meaningful.
- **Class-label switching** — sort classes by a monotone characteristic (intercept, slope) after fitting for reproducibility.
- **Overextraction risk** — LGMM will find classes even when the data are homogeneous but not exactly Gaussian; report residual and posterior-class-probability diagnostics.

## Run

```
python techniques/latent-growth-mixture/python/latent_growth_mixture.py
Rscript techniques/latent-growth-mixture/r/latent_growth_mixture.R
```

**Refs:** Nagin, D.S. "Analyzing developmental trajectories: a semiparametric, group-based approach." *Psychol. Methods* 4(2), 139–157, 1999; Muthén, B. "Latent variable analysis: growth mixture modeling and related techniques for longitudinal data." *Handbook of Quantitative Methodology for the Social Sciences* 345–368, 2004; Proust-Lima, C., Philipps, V. & Liquet, B. "Estimation of extended mixed models using latent classes and latent processes: the R package lcmm." *J. Stat. Softw.* 78(2), 2017.

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
