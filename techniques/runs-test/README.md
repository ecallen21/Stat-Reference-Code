# Wald-Wolfowitz Runs Test (Reference §7.15)

A **run** is a maximal subsequence of identical elements. Given a binary sequence with `n_1` A's and `n_2` B's, count the total number of runs `R`. Under the null (sequence is a **random ordering**),

```
μ_R  = 2 n_1 n_2 / (n_1 + n_2) + 1
σ_R² = 2 n_1 n_2 (2 n_1 n_2 − n_1 − n_2) / ((n_1 + n_2)² (n_1 + n_2 − 1))
z    = (R − μ_R) / σ_R    ~ N(0, 1)
```

- **Too few runs** → clustering / streakiness.
- **Too many runs** → over-alternation.

## Continuous data

Dichotomize around the median (or another cutoff) → runs test on the above/below labels. Useful for testing whether residuals of a regression fit are randomly scattered vs show serial correlation / trend.

## Files

- `python/runs_test.py` — from-scratch runs counter + Normal-approximation p-value + auto-dichotomization for continuous input. Demos:
  - Random 100 binary: z = −1.89, p = 0.06 (fail to reject).
  - Fully clustered (all 0s then all 1s): z = −9.85, p ≈ 0 (reject).
  - Fully alternating (0101...): z = +9.85, p ≈ 0 (reject).
  - Continuous residuals with a slight trend: z = −3.38, p = 0.0007 (reject).

- `r/runs_test.R` — `tseries::runs.test` and `randtests::runs.test`.

## When to use

- **Testing randomness** of a binary sequence — coin flips, quality-control pass/fail, Bernoulli trials.
- **Residual diagnostics** — after regression / time-series fit, dichotomize residuals at the median and run the test; significant clustering suggests unmodelled serial dependence.
- **Quality-control** — CUSUM alternative for detecting runs of out-of-control values.

## Assumptions & caveats

- **Two categories only** — for polytomous or continuous data without an obvious cutoff, prefer other tests (Ljung-Box for autocorrelation, KS, etc.).
- **Asymptotic Normal p-value** — for small `min(n_1, n_2) < 10`, use the exact distribution (permutation).
- **Independence of trials** under the null — the test detects deviations from independence.

## Related methods

- **Autocorrelation / Ljung-Box** (see `acf-pacf`) — better for continuous serial data.
- **Kolmogorov-Smirnov** — different null (distribution shape).
- **Mood test** — dispersion under a location shift.

## Run

```
python techniques/runs-test/python/runs_test.py
Rscript techniques/runs-test/r/runs_test.R
```

**Refs:** Wald, A. & Wolfowitz, J. "On a test whether two samples are from the same population." *Ann. Math. Stat.* 11(2), 147–162, 1940; Bradley, J.V. *Distribution-Free Statistical Tests*, Prentice-Hall, 1968.

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
