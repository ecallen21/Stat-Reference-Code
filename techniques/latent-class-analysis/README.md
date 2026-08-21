# Latent Class Analysis — LCA (Reference §19.x extra)

Categorical-latent-variable model for **categorical indicators**. Assumes a
discrete latent class `C_i ∈ {1, …, K}` that fully explains associations
between observed indicators (local independence):

```
P(U_i = u) = Σ_k π_k · Π_j p_{jk}^{u_j} · (1 − p_{jk})^{1 − u_j}
```

- `π_k` = prior class prevalence.
- `p_{jk}` = probability of item `j` positive given class `k`.
- `C_i` classifies each subject probabilistically via Bayes' rule after fitting.

## Fitting: EM

- **E**: `γ_{ik} = π_k · P(U_i | C = k) / Σ_l π_l · P(U_i | C = l)`
- **M**: `π_k = mean_i γ_{ik}`; `p_{jk} = Σ_i γ_{ik} · U_{ij} / Σ_i γ_{ik}`

Iterate to convergence. Use multiple random restarts — LCA has many local
optima, especially with many classes.

## Choosing K

Common criteria:

- **BIC** — most-used default; parsimonious.
- **AIC** — larger K tolerated; risk of over-extraction.
- **Bootstrap Likelihood Ratio Test (BLRT)** — Nylund et al. 2007; often best but expensive.
- **Entropy** — class-assignment separation quality (higher = better).
- **Substantive interpretability** — always required.

## Contrast with related methods

- **Latent Profile Analysis (LPA)** — continuous indicators (Gaussian conditional).
- **Latent Transition Analysis (LTA)** — LCA over waves; adds transition probabilities.
- **Factor Mixture Models** — hybrid categorical + continuous latent structure.
- **Growth Mixture Models** — trajectories rather than static classes (see `latent-growth-mixture`).
- **Finite mixture of regressions** — different regression coefficients per class.

## When to use

- **Typology / clustering with categorical indicators** — patient-reported outcomes, survey items, diagnostic checklists.
- **Model-based cluster analysis** — LCA is the categorical analogue of Gaussian mixture / GMM.
- **Measurement of unobserved subgroups** — depression subtypes, response styles, latent segments.
- **Data reduction** — replace many correlated categorical items with a single class assignment.

## Files

- `python/latent_class_analysis.py` — from-scratch EM with multiple restarts + BIC / AIC selection. Demo (n=800, J=6, true K=3 with well-separated response patterns): BIC minimum at K=3 (5353.7 vs K=2's 5361.0); estimated prevalences π = [0.39, 0.32, 0.29] vs true [0.40, 0.35, 0.25]; class response probabilities recovered to within 0.02–0.03.
- `r/latent_class_analysis.R` — `poLCA::poLCA` (canonical), `depmixS4::mix`, `Mclust` (continuous LPA), Python `stepmix`.

## Assumptions & caveats

- **Local independence** — items are conditionally independent given class. Residual associations (correlated items sharing a stem) inflate K estimates; check with residual correlations after fit.
- **Local maxima** — always use 20+ random restarts; report the log-lik at the best solution alongside others.
- **Class label switching** across runs — align classes by matching response-probability profiles before comparison.
- **Small classes** — a class of < ~5% of the sample is often unstable; consider merging.
- **Direct effects of covariates** — if a covariate influences an item over-and-above the class, use LCA-with-covariates (3-step approach, Vermunt 2010) rather than a naive 1-step model.
- **N × K × J** — EM cost is linear in each; fits with K > 8 and J > 30 can be slow to converge.

## Related in this repo

- `gaussian-mixture-models` — continuous analogue.
- `dirichlet-process-mixture` — Bayesian alternative that infers K.
- `hmm` — LCA extended to sequences (latent-transition).
- `group-based-trajectory`, `latent-growth-mixture` — trajectory-class analogues.

## Run

```
python techniques/latent-class-analysis/python/latent_class_analysis.py
Rscript techniques/latent-class-analysis/r/latent_class_analysis.R
```

**Refs:** Goodman, L.A. "Exploratory latent structure analysis using both identifiable and unidentifiable models." *Biometrika* 61(2), 215–231, 1974; Collins, L.M. & Lanza, S.T. *Latent Class and Latent Transition Analysis*, Wiley, 2010; Nylund, K.L., Asparouhov, T. & Muthén, B.O. "Deciding on the number of classes in latent class analysis and growth mixture modeling: A Monte Carlo simulation study." *Struct. Equ. Modeling* 14(4), 535–569, 2007.

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
