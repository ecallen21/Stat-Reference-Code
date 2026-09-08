# Bai-Perron Multiple Structural Breaks (Reference §12.17)

Bai & Perron (1998, 2003). Detects **multiple unknown break points**
in a linear regression:

    y_t = X_t · β_j + ε_t     for t ∈ (τ_{j−1}, τ_j]  (j = 1, …, m+1)

Efficient dynamic-programming search finds global least-squares
break locations in `O(T²)`; sup-F, UDMax, WDMax tests + BIC choose
the number of breaks `m`.

## Files

- `python/bai_perron_multiple_breaks.py` — DP for a given `m` +
  BIC across `m ∈ {0, 1, 2, 3}` from scratch. Demo (T=200, true
  breaks at 70, 140): BIC picks m = 2 with estimated breaks
  (69, 139) — recovered to ±1 index.
- `r/bai_perron_multiple_breaks.R` — `strucchange::breakpoints`,
  `changepoint::cpt.mean` (R); `ruptures`, from-scratch (Python).

## When to use

- **Time-series regression** where you suspect multiple regime
  shifts.
- **Post-hoc regime detection** — inflation targeting, monetary
  policy, market structure changes.
- **Building an ITS specification** — determine break count
  before fitting piecewise ARIMA / linear models.

## When NOT to use

- **Continuous / smooth drift** — use time-varying-coefficient
  models or splines.
- **Very small T** — few observations per segment; DP unstable.
- **Non-stationary variance** — sensitivity to break in σ² too;
  extend to double-break variance models.

## Assumptions & caveats

- **Minimum segment length** `h` — usually 5-15 % of `T` to avoid
  spurious breaks near boundaries.
- **Homoskedastic vs HAC-robust** — sup-F variants have both
  versions; strucchange defaults to HAC.
- **BIC vs sequential sup-F** — BIC tends to under-select breaks;
  sup-F selects on p-values (Bai-Perron argue against).
- **Serial correlation** — break dates unbiased but SE inflated;
  use HAC-adjusted CIs on break dates.

## Related in this repo

- `chow-test-structural-break`, `change-point-detection`,
  `structural-breaks-its` — related break-detection cousins.
- `regime-switching-markov` — model-based regime alternative.
- `event-study`, `staggered-did` — event-driven analogues.

## Run

```
python techniques/bai-perron-multiple-breaks/python/bai_perron_multiple_breaks.py
Rscript techniques/bai-perron-multiple-breaks/r/bai_perron_multiple_breaks.R
```

**Refs:** Bai, J. & Perron, P. "Estimating and testing linear models with multiple structural changes." *Econometrica*, 66(1): 47-78, 1998; Bai, J. & Perron, P. "Computation and analysis of multiple structural change models." *Journal of Applied Econometrics*, 18(1): 1-22, 2003.

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
