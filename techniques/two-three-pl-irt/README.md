# 2PL and 3PL Item Response Theory (Reference §22.6)

Extensions of the Rasch (1PL) model that let items differ in discrimination and (for 3PL) guessing.

## 2PL (Birnbaum 1968)

```
Pr(y_ij = 1 | θ_i, a_j, b_j) = 1 / (1 + exp(−a_j (θ_i − b_j)))
```

- `a_j` — **discrimination** (slope): how sharply the probability rises around `b_j`.
- `b_j` — difficulty.

## 3PL (Birnbaum / Lord 1980)

```
Pr(y_ij = 1 | ...) = c_j + (1 − c_j) · σ(a_j (θ_i − b_j))
```

- `c_j` — pseudo-guessing lower asymptote. 4-option MC: `c_j ≈ 0.25`.

## Marginal MLE (MML) estimation

Integrate `θ ~ N(0, 1)` via Gauss-Hermite quadrature:

```
log-lik = Σ_i log ∫  Π_j Pr(y_ij | θ) φ(θ) dθ
```

Optimize `(a, b, [c])` by BFGS. Person `θ` recovered post-fit by **EAP** (expected a posteriori):

```
θ̂_i = ∫ θ · Pr(θ | y_i) dθ
```

## Files

- `python/two_three_pl_irt.py` — from-scratch 2PL MML with 15-node Gauss-Hermite + EAP theta estimator. Demo (n = 400, J = 12): correlation of estimated `b` with truth = 0.99; correlation of `a` with truth = 0.83.
- `r/two_three_pl_irt.R` — pointers to `ltm::ltm`, `ltm::tpm`, `mirt::mirt(itemtype = "2PL" / "3PL")`.

## When to use each

- **2PL** — items differ substantively in how sharply they distinguish ability. Standard in educational and psych measurement.
- **3PL** — multiple-choice items where low-ability examinees can guess. Adds identifiability challenges (`c_j` needs a lot of data).
- **1PL / Rasch** (`rasch-model`) — when items should be treated as equally discriminating and additivity of the scale matters.

## Model selection

- Compare 1PL vs 2PL vs 3PL by likelihood-ratio test or AIC/BIC.
- Item-fit statistics (S-χ², Q1) flag poor-fitting items.
- Person-fit statistics (see `person-fit-statistics`) flag aberrant response patterns.

## Assumptions & caveats

- **Unidimensionality** — one latent trait `θ`.
- **Local independence** — given `θ`, item responses are independent.
- **Sample size** — 500+ for 2PL, 1000+ for 3PL (three parameters per item).
- **Guessing parameter `c_j`** is weakly identified — use Bayesian priors (e.g. Beta(5, 17)) or fix at a plausible value.

## Run

```
python techniques/two-three-pl-irt/python/two_three_pl_irt.py
Rscript techniques/two-three-pl-irt/r/two_three_pl_irt.R
```

**Refs:** Birnbaum, A. "Some latent trait models and their use in inferring an examinee's ability." In F.M. Lord & M.R. Novick, *Statistical Theories of Mental Test Scores*, Addison-Wesley, 1968; Lord, F.M. *Applications of Item Response Theory to Practical Testing Problems*, Lawrence Erlbaum, 1980; Chalmers, R.P. "mirt: A multidimensional item response theory package for the R environment." *J. Stat. Softw.* 48(6), 1–29, 2012.

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
