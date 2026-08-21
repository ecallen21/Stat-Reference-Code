# Person-Fit Statistics (Reference §22.13)

Detect **aberrant** examinee response patterns given estimated IRT item parameters. Uses the model to flag patterns that are too unlikely to have come from any single latent ability.

## Drasgow-Levine-Williams l_z (1985)

```
ℓ     = Σ_j y_j log P_j + (1 − y_j) log(1 − P_j)
E[ℓ]  = Σ_j P_j log P_j + (1 − P_j) log(1 − P_j)
Var[ℓ] = Σ_j P_j (1 − P_j) (log(P_j / (1 − P_j)))²
l_z   = (ℓ − E[ℓ]) / √Var[ℓ]                          ~ N(0, 1) approximately
```

## Interpretation

- **l_z << −2** — pattern too unlikely; possible cheating, random responding, disengagement, or misunderstood instructions (person gets easy items wrong and hard items right).
- **l_z >> +2** — pattern too good to be true (rare but flagged for review).

## Files

- `python/person_fit_statistics.py` — from-scratch l_z + MLE θ per person given item parameters. Demo (n = 300, J = 20, 2 aberrant persons planted): inverted person l_z = −6.5, random-responder l_z = −2.5; overall distribution mean 0.08, sd 0.97 (approximately standard normal).
- `r/person_fit_statistics.R` — `PerFit::lz` / `PerFit::HT` / `PerFit::G`; `mirt::personfit`.

## When to use

- **Large-scale operational testing** — routine screening to flag suspicious responses (SAT, GRE, medical licensure).
- **Adaptive testing** — detect response aberrance in real time.
- **Post-hoc quality control** — assess whether estimated abilities are trustworthy for reporting.

## Assumptions & caveats

- **Requires estimated θ and item parameters** — bootstrap or Bayesian intervals give more principled thresholds.
- **Test length matters** — short tests give unreliable l_z; use ≥ 20 items for standard-normal approximation.
- **Multiple comparisons** — flagging many respondents in a large operational cohort needs correction (BH-FDR).
- **Interpretation is diagnostic**, not diagnostic-of-fraud — always follow up with content review.

## Related statistics

- **Zh** (Snijders 2001) — corrects l_z's bias when θ is estimated.
- **H_T** (Sijtsma-Meijer) — normed conformity index.
- **infit / outfit** — Rasch-family fit statistics per person.
- **G, U, C** — various response-consistency indices; comprehensive coverage in the `PerFit` R package.

## Run

```
python techniques/person-fit-statistics/python/person_fit_statistics.py
Rscript techniques/person-fit-statistics/r/person_fit_statistics.R
```

**Refs:** Drasgow, F., Levine, M.V. & Williams, E.A. "Appropriateness measurement with polychotomous item response models and standardized indices." *Br. J. Math. Stat. Psychol.* 38(1), 67–86, 1985; Snijders, T.A.B. "Asymptotic null distribution of person-fit statistics with estimated person parameter." *Psychometrika* 66(3), 331–342, 2001.

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
