# Samejima's Graded Response Model (Reference §22.7)

Extension of 2PL to **ordered polytomous** items (Likert scales, rubric ratings, symptom-severity). Samejima 1969.

## Model

For an item with `K` ordered categories `0, 1, ..., K−1`, define `K−1` **cumulative probabilities** of scoring at least `k`:

```
P_j^*(k | θ) = 1 / (1 + exp(−a_j (θ − b_jk))),   k = 1, ..., K−1
P_j^*(0 | θ) = 1
P_j^*(K | θ) = 0
```

Category probability:

```
P_j(k | θ) = P_j^*(k | θ) − P_j^*(k + 1 | θ)
```

Thresholds automatically ordered by the logistic monotone form (`b_j1 < b_j2 < ... < b_j,K−1`).

## Files

- `python/graded_response_model.py` — from-scratch MML with Gauss-Hermite quadrature + softplus reparameterization to enforce ordered thresholds. Demo (n = 500, J = 5, K = 4): correlation of estimated `a` with truth = 0.72; correlation of thresholds with truth = 0.99.
- `r/graded_response_model.R` — `ltm::grm` or `mirt::mirt(Y, 1, itemtype = "graded")`.

## When to use

- **Likert-scale items** (1–5 or 1–7 ratings) — most psychometric surveys.
- **Rubric-scored** classroom / clinical items with a few ordered grades.
- **PROMIS instruments** and other IRT-based patient-reported outcomes use GRM extensively.

## Contrast with alternatives

- **Partial Credit Model** (`partial-credit-model`, Masters) — different parameterization; equally-spaced thresholds a special case.
- **Generalized Partial Credit** (Muraki) — PCM + discrimination.
- **Rating Scale Model** — same category thresholds across items.
- **Nominal Response Model** (Bock) — unordered categorical items.

## Assumptions & caveats

- **Ordered categories** — never use GRM for nominal items.
- **Sample size** — 300+ for reasonable stability; more items ⇒ better than more people typically.
- **Local independence** — items independent given `θ`.
- **Missing categories** — a threshold that no examinees crossed becomes unestimable; collapse or exclude.

## Run

```
python techniques/graded-response-model/python/graded_response_model.py
Rscript techniques/graded-response-model/r/graded_response_model.R
```

**Refs:** Samejima, F. "Estimation of latent ability using a response pattern of graded scores." *Psychometrika Monograph Supplement* 17, 1969; Rizopoulos, D. "ltm: An R package for latent variable modeling and item response analysis." *J. Stat. Softw.* 17(5), 1–25, 2006.

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
