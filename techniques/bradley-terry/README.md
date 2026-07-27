# Bradley–Terry Model for Pairwise Comparisons (Reference §8.8)

Given items 1..K and a record of who beat whom in pairwise matchups (games, votes, preferences), the Bradley–Terry model assigns each item a positive **ability** `π_i` such that

```
P(i beats j)  =  π_i / (π_i + π_j)
```

Equivalent logistic form with `β_i = log(π_i)`:

```
logit P(i beats j)  =  β_i − β_j
```

with the identifiability constraint `Σ β_i = 0` (equivalently, geometric-mean(π) = 1).

## Fitting: MM algorithm

Ford (1957) / Hunter (2004) give a monotone-convergent, derivative-free update:

```
π_i  ←  W_i  /  Σ_{j ≠ i} n_ij / (π_i + π_j)
```

where `W_i` = wins by i and `n_ij` = games between i and j. Re-normalize to geometric mean 1 each iteration.

**Wald SEs**: from the Fisher information for β, whose entries are

```
J_ii = Σ_{j ≠ i}  n_ij π_i π_j / (π_i + π_j)²
J_ij = −n_ij π_i π_j / (π_i + π_j)²      (i ≠ j)
```

with the pseudoinverse handling the sum-to-zero constraint.

## What you get

- `π`: ability on multiplicative scale (double it → other item's odds of beating you halve, all else equal).
- `β = log π`: additive scale; ready for a Wald test of `β_i − β_j = 0` (equal skill).
- `ranking_descending`: items sorted best → worst.
- `predict_win_prob(fit, i, j)`: probability i beats j under the fit.

## Files

- `python/bradley_terry.py` — MM fitter + Wald SEs + prediction helper; optional cross-check against `choix.ilsr_pairwise` when installed.
- `r/bradley_terry.R` — from-scratch MM + `BradleyTerry2::BTm` when available.

## Assumptions

- Pairwise comparisons only (no ties by default — Davidson 1970 extends the model to handle ties; not implemented here).
- **Strong connectivity**: the "who beat whom" graph must be connected via directed edges — if item X never won and never lost, its π isn't identified.
- Independence across matchups (no fatigue, no order effects).

## Run

```
python techniques/bradley-terry/python/bradley_terry.py
Rscript techniques/bradley-terry/r/bradley_terry.R
```

**Refs:** Bradley, R.A. & Terry, M.E. "Rank analysis of incomplete block designs: I. The method of paired comparisons." *Biometrika* 39(3–4), 324–345, 1952; Ford, L.R. "Solution of a ranking problem from binary comparisons." *Am. Math. Monthly* 64(8), 28–33, 1957; Hunter, D.R. "MM algorithms for generalized Bradley-Terry models." *Ann. Stat.* 32(1), 384–406, 2004.

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
