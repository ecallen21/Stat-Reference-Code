# Capture-Recapture (Reference §38.11)

Chao et al. (2001), Otis et al. (1978). Estimate an unknown
population size `N` from overlapping incomplete lists / capture
occasions.

## Two-sample

- **Lincoln-Petersen** — `N̂ = n₁ · n₂ / m`; simple, biased for
  small `m`.
- **Chapman (1951)** — `N̂ = (n₁+1)(n₂+1)/(m+1) − 1`; bias-
  corrected, closed-form variance
  `Var = (n₁+1)(n₂+1)(n₁−m)(n₂−m) / ((m+1)² (m+2))`.

## Multi-sample

- **Schnabel** — `N̂ = Σ_t n_t M_t / Σ_t m_t`, with `M_t = |seen
  before t|`, `m_t = |recaptured at t|`.
- **Log-linear** (Otis et al. 1978) — model heterogeneity in
  capture probability across time / behaviour / individuals.

## When to use

- **Wildlife abundance** — the classical mark-recapture literature.
- **Epidemiology** — case ascertainment across registries (disease
  burden, homelessness counts).
- **Undercount adjustment** in census / survey work.

## When NOT to use

- **Assumptions grossly violated** — open populations
  (births/deaths/migration), lost marks, heterogeneous or
  behavioural capture response.
- **Only one list** — nothing to overlap; consider chart-review
  gold-standard studies instead.

## Files

- `python/capture_recapture.py` — Lincoln-Petersen, Chapman (with
  SE), Schnabel. Demo (N=400, p=0.25, T=5): two-occasion
  Chapman **N̂ = 484.8, 95%CI (324, 646)**; five-occasion Schnabel
  **N̂ = 424.5**.
- `r/capture_recapture.R` — `Rcapture`, `CARE1`, `multimark`,
  `secr` (R); custom (Python).

## Assumptions & caveats

- **Closure** — no births / deaths / migration during the study
  (Jolly-Seber relaxes this).
- **Equal capture probability** — heterogeneity biases `N̂`
  downward; log-linear models with covariates help.
- **Independent captures** — behavioural response (trap-happy /
  trap-shy) violates this.
- **Marks retained + identifiable** — lost tags inflate `N̂`.
- **Small `m`** — even Chapman's bias correction breaks down when
  `m ≤ 2`.

## Related in this repo

- `mark-resight` (if present) — extension when re-encounters are
  observation-only.
- `bayesian-glms` — Bayesian alternatives put priors on capture
  probability.

## Run

```
python techniques/capture-recapture/python/capture_recapture.py
Rscript techniques/capture-recapture/r/capture_recapture.R
```

**Refs:** Chapman, D.G. "Some properties of the hypergeometric distribution with applications to zoological sample censuses." *Univ Calif Publ Stat*, 1951; Otis, D.L., Burnham, K.P., White, G.C., & Anderson, D.R. "Statistical inference from capture data on closed animal populations." *Wildlife Monographs*, 1978; Chao, A. et al. "The applications of capture-recapture models to epidemiological data." *Statistics in Medicine*, 2001.

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
