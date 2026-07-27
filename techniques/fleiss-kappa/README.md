# Fleiss' Kappa: Agreement Among m ≥ 3 Raters (Reference §8.4)

Extends Cohen's κ (two raters) to any number of raters and does NOT require the same raters to score every item. What it *does* require is that every item is scored by the same number `m` of raters.

## The input

An `n × K` matrix — one row per item, one column per category — where cell `(i, j)` is *the number of raters who assigned item i to category j*. Every row sums to `m`.

## The formula

```
P_i    = (1 / m(m−1)) · Σ_j n_ij (n_ij − 1)     (per-item raw agreement)
P̄     = mean_i P_i                              (overall observed agreement)
p_j    = Σ_i n_ij / (n · m)                     (marginal category rate)
P_e    = Σ_j p_j²                                (chance agreement)
κ      = (P̄ − P_e) / (1 − P_e)
```

`ASE` (Fleiss 1971), Wald z-test, and per-category κ_j are also produced.

## Files

- `python/fleiss_kappa.py` — from-scratch κ, ASE, per-category κ_j; matches `statsmodels.stats.inter_rater.fleiss_kappa` to 12 decimals.
- `r/fleiss_kappa.R` — from-scratch + `irr::kappam.fleiss`.

## Assumptions

- Nominal categories (for ordinal, agreement structure is better handled with `weighted-kappa`).
- Same number of raters per item (`m` constant).
- Rater identity does NOT matter — Fleiss treats raters as exchangeable. If your raters have fixed identities and you want to attribute disagreement to specific pairings, use pairwise Cohen's κ instead.

## Run

```
python techniques/fleiss-kappa/python/fleiss_kappa.py
Rscript techniques/fleiss-kappa/r/fleiss_kappa.R
```

**Refs:** Fleiss, J.L. "Measuring nominal scale agreement among many raters." *Psychological Bulletin* 76(5), 378–382, 1971; Fleiss, J.L., Levin, B. & Paik, M.C. *Statistical Methods for Rates and Proportions*, 3rd ed., Wiley, 2003 (Ch. 18).

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
