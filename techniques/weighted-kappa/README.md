# Weighted Kappa for Ordinal Agreement (Reference §8.4)

Cohen's κ treats every disagreement as equally bad. For **ordinal** categories that's wrong — a rater who says "moderate" when the truth is "mild" is closer than one who says "critical". Weighted κ encodes this via a K×K weight matrix `W` with `W_ii = 0` (perfect agreement) and `W_ij` increasing in `|i − j|`.

## Weight schemes

For K ordered categories `1..K`:

- **Linear** (Cicchetti–Allison): `W_ij = |i − j| / (K − 1)` — penalty proportional to distance.
- **Quadratic** (Fleiss–Cohen): `W_ij = (i − j)² / (K − 1)²` — penalty proportional to *squared* distance. Weighted κ with quadratic weights equals the intraclass correlation coefficient ICC(2,1) on the ratings treated as scores — this is why medicine uses it as *the* IRR statistic.

## Formula

```
κ_w  =  1  −  Σ Σ W_ij · p_ij  /  Σ Σ W_ij · p_i· · p_·j
             (observed weighted disagreement / chance-weighted disagreement)
```

## SE / CI

The Fleiss–Cohen–Everitt delta-method ASE is well-known to be **fragile** — for typical data (most weight concentrated on the diagonal) it suffers numerical cancellation and can collapse to zero. This implementation uses a **nonparametric bootstrap** SE and percentile CI instead — robust in all regimes and produces the same numbers you'd get from `boot::boot()` in R.

## Files

- `python/weighted_kappa.py` — from-scratch `κ_w` (both weight schemes), bootstrap SE, percentile CI. Point estimates match `sklearn.metrics.cohen_kappa_score(weights=…)` to 12 decimals for both schemes.
- `r/weighted_kappa.R` — from-scratch + `irr::kappa2(weight="equal"|"squared")`.

## Assumptions

- Two raters, K **ordered** categories in the natural sequence.
- The weight scheme should match how substantively bad off-by-k disagreements are for your problem — quadratic is standard when a large gap is *much* worse than a small one.

## Run

```
python techniques/weighted-kappa/python/weighted_kappa.py
Rscript techniques/weighted-kappa/r/weighted_kappa.R
```

**Refs:** Cohen, J. "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit." *Psych. Bull.* 70(4), 213–220, 1968; Fleiss, J.L. & Cohen, J. "The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability." *Educ. Psych. Meas.* 33(3), 613–619, 1973; Cicchetti, D.V. & Allison, T. "A new procedure for assessing reliability of scoring EEG sleep recordings." *Am. J. EEG Tech.* 11(3), 101–110, 1971.

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
