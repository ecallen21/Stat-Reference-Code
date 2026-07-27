# Subsampling and m-out-of-n Bootstrap (Reference §10.10; also covers §10.15)

Two resampling schemes that use **subsamples of size `m < n`** instead of full-size samples:

| Method | Sampling | Ref |
|---|---|---|
| **Subsampling** (Politis–Romano–Wolf 1999) | Without replacement | §10.10 |
| **m-out-of-n bootstrap** (Bickel–Götze–van Zwet 1997) | With replacement | §10.15 |

## Why bother?

For most statistics the ordinary n-out-of-n bootstrap works fine. But there are known cases where it's **inconsistent** (does not converge to the correct limiting distribution):

- Extreme-value statistics (`max`, `min`, `range`).
- Statistics at boundaries of the parameter space (e.g. variance under a null of zero).
- Unit-root time series.
- Some non-smooth estimators.

Under mild regularity, subsampling and the m-out-of-n bootstrap remain **consistent** in these pathological cases. Choose `m` such that `m → ∞` and `m/n → 0` as `n → ∞`. A common rule of thumb is `m ≈ √n` or `m ≈ n^(2/3)`.

## Demo — sample maximum (from the Python demo)

The exponential(1) sample max is at 5.01. Bootstrap CIs at n = 200:

| Method | 95% CI |
|---|---|
| Naive n-out-of-n bootstrap | [3.90, **5.01**] ← stuck at observed max — inconsistency signature |
| m-out-of-n bootstrap, m = 50 | [2.68, 5.01] |
| Subsampling, m = 50 | [5.01, **6.09**] ← correctly extends above the observed max |

Subsampling with the rate scaling `a_n = √n` inverts the correct limiting distribution and produces a CI that acknowledges the true underlying max is likely larger than the observed one.

## Files

- `python/subsampling.py` — both variants with the sqrt-n rate scaling default; demo on the sample max.
- `r/subsampling.R` — from-scratch versions.

## Assumptions

- `m` grows with `n` but slowly (`m/n → 0`). Otherwise you're just doing the ordinary bootstrap in disguise.
- Rate scaling `a_n` must match the true convergence rate of the statistic. Default `√n` is right for most smooth statistics; extreme-value or boundary problems need a different exponent.

## Run

```
python techniques/subsampling/python/subsampling.py
Rscript techniques/subsampling/r/subsampling.R
```

**Refs:** Politis, D.N., Romano, J.P. & Wolf, M. *Subsampling*, Springer, 1999; Bickel, P.J., Götze, F. & van Zwet, W.R. "Resampling fewer than n observations: gains, losses, and remedies for losses." *Stat. Sinica* 7(1), 1–31, 1997.

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
