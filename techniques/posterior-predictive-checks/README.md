# Posterior Predictive Checks (Reference §14.19)

Diagnostic for Bayesian model fit. Simulate replicated datasets `y_rep` from the posterior predictive

```
y_rep ~ p(y_rep | y) = ∫ p(y_rep | θ) p(θ | y) dθ
```

For each posterior draw `θ^(s)`, simulate a replicated dataset of the same size as `y`, compute a summary statistic `T(y_rep^(s))`, and compare its distribution to `T(y)`.

## Bayesian p-value

```
p_B = Pr(T(y_rep) ≥ T(y) | y)
```

- `p_B ≈ 0.5` — the model reproduces `T` well.
- `p_B ≈ 0` or `≈ 1` — the model systematically misses `T`; flag as misfit.

## Choosing test statistics

- Pick statistics the model was **not designed to fit**. Testing `T = mean` on a Normal-mean model is nearly tautological and never rejects.
- **Distributional shape**: skewness, kurtosis, quantiles.
- **Extremes**: min, max, tail quantiles — Normal models miss heavy tails badly.
- **Structural**: number of zeros in a count model, lag-1 autocorrelation in a TS model, proportion above a clinical threshold.

## Files

- `python/posterior_predictive_checks.py` — from-scratch PPC runner reporting per-statistic Bayesian p-values and 95% simulation intervals. Demos show a correctly-specified Normal-on-Normal fit (all p-values near 0.5) and a mis-specified Normal-on-Cauchy fit (kurtosis, min, max flagged at p ≈ 0 or 1).
- `r/posterior_predictive_checks.R` — same summary table in base R; production alternative is `bayesplot::ppc_dens_overlay` / `bayesplot::ppc_stat`.

## When to use

- After any Bayesian model fit, before reporting parameter estimates.
- Comparing candidate likelihoods (Normal vs Student-t; Poisson vs Negative-Binomial).
- Sanity-checking model expansions (does adding random slopes fix the misfit on lag-1 autocorrelation?).

## Caveats

- PP p-values are **not** frequentist p-values — they can be conservative (near 0.5 even for wrong models) or anti-conservative depending on the statistic.
- Use PPCs alongside WAIC / LOO (predictive score) and posterior-predictive **density overlays** (visual check).
- If nothing rejects, the model may still be wrong in ways your chosen statistics don't reveal.

## Run

```
python techniques/posterior-predictive-checks/python/posterior_predictive_checks.py
Rscript techniques/posterior-predictive-checks/r/posterior_predictive_checks.R
```

**Refs:** Gelman, A., Meng, X.-L. & Stern, H. "Posterior predictive assessment of model fitness via realized discrepancies." *Stat. Sinica* 6(4), 733–807, 1996; Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., CRC, 2013 (Ch 6).

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
