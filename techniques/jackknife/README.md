# Jackknife: SE, Bias Correction, and Jackknife-After-Bootstrap (Reference §10.6; also covers §10.17)

The **jackknife** is the pre-bootstrap resampling procedure: compute the statistic on each leave-one-out subsample and use the spread of those replicates.

## Definitions

For sample `x₁..x_n` and statistic `θ̂ = T(x)`:

```
J_i    =  T(x₁, ..., x_{i−1}, x_{i+1}, ..., x_n)     for i = 1..n
J̄     =  mean of J_i

SE_jack    =  √( (n − 1)/n · Σᵢ (J_i − J̄)² )
bias_hat   =  (n − 1) · (J̄ − θ̂)
θ_BC       =  θ̂ − bias_hat  =  n·θ̂ − (n − 1)·J̄
```

## Bootstrap vs. jackknife

| | Jackknife | Bootstrap |
|---|---|---|
| Replicates | `n` (deterministic) | `B` (random) |
| Cost | Cheap for small n | Scales with B |
| Non-smooth statistics (e.g. median) | Poor (LOO too coarse) | Fine |
| Skewness / higher moments | First-order only | Can capture |
| Bias correction | Built-in formula | Bootstrap bias also available |

Rule of thumb: jackknife when the statistic is smooth and `n` is small; bootstrap otherwise.

## Jackknife-after-bootstrap (Efron 1992)

Diagnostic: measures the **influence of each observation on the bootstrap SE**. For each `i`, compute the bootstrap SE of `θ*` restricted to those bootstrap samples that don't contain obs `i`. Big differences from the full bootstrap SE identify influential / leverage points.

## Files

- `python/jackknife.py` — 1-D and 2-D jackknife SE + bias correction + jackknife-after-bootstrap. Demo: bias correction on the biased sample variance (÷n) recovers the unbiased sample variance (÷n−1) **exactly**.
- `r/jackknife.R` — from-scratch + `bootstrap::jackknife` when installed.

## Assumptions

- Statistic is **smooth** in the observations. For medians/quantiles the LOO changes are too discrete; use bootstrap SEs there.
- Independent observations; for dependent data, use block jackknife (not implemented here — extension of block-bootstrap ideas).

## Run

```
python techniques/jackknife/python/jackknife.py
Rscript techniques/jackknife/r/jackknife.R
```

**Refs:** Quenouille, M.H. "Notes on bias in estimation." *Biometrika* 43(3/4), 353–360, 1956; Tukey, J.W. "Bias and confidence in not-quite large samples." *Ann. Math. Stat.* 29, 614, 1958 (abstract); Efron, B. "Jackknife-after-bootstrap standard errors and influence functions." *JRSS B* 54(1), 83–127, 1992; Miller, R.G. "The jackknife — a review." *Biometrika* 61(1), 1–15, 1974.

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
