# Partial Credit + Generalized Partial Credit (Reference §22.8)

Rasch-family IRT for **polytomous** items where scoring credits intermediate steps.

## Partial Credit Model (Masters 1982)

For an item with `K` ordered categories, the probability of category `k` is:

```
P_j(k | θ) = exp( Σ_{h=0}^{k} (θ − δ_jh) )
            / Σ_{k'=0}^{K−1} exp( Σ_{h=0}^{k'} (θ − δ_jh) )
```

with `δ_j0 = 0`. Each `δ_jh` is the **step difficulty** from category `h − 1` to `h`. Unlike GRM (which forces ordered thresholds), PCM's step difficulties **need not be ordered** — reversals happen when a middle category is locally hardest to reach.

## Generalized PCM (Muraki 1992)

Adds a discrimination `a_j`:

```
numerator = exp(a_j · Σ_{h=0}^{k} (θ − δ_jh))
```

Reduces to PCM when `a_j = 1`.

## Files

- `python/partial_credit_model.py` — from-scratch GPCM MML with Gauss-Hermite quadrature. Demo (n = 500, J = 4, K = 4): correlation of `δ` with truth = 0.997; correlation of `a` with truth = 0.90.
- `r/partial_credit_model.R` — pointers to `ltm::gpcm`, `mirt::mirt(itemtype = "gpcm" / "PCM")`, `eRm::PCM` (CML).

## PCM vs GRM

|                | PCM / GPCM                                    | GRM (Samejima)                          |
|----------------|------------------------------------------------|-----------------------------------------|
| Parameterization | step difficulties (adjacent-category logit)  | thresholds (cumulative logit)           |
| Reversals      | allowed (informative for hardest step)        | precluded by monotone P*                |
| Best for       | scoring rubrics where partial credit is real  | Likert scales with ordered thresholds   |
| Rasch member   | yes (PCM); GPCM adds discrimination           | no                                      |

## Assumptions & caveats

- **Ordered categories** — never for nominal responses.
- **Local independence + unidimensionality**.
- **Small categories** — collapse rarely-used adjacent categories to stabilize estimation.
- **Interpret step reversals** substantively — sometimes an indicator of poor category definitions.

## Run

```
python techniques/partial-credit-model/python/partial_credit_model.py
Rscript techniques/partial-credit-model/r/partial_credit_model.R
```

**Refs:** Masters, G.N. "A Rasch model for partial credit scoring." *Psychometrika* 47(2), 149–174, 1982; Muraki, E. "A generalized partial credit model: application of an EM algorithm." *Appl. Psychol. Meas.* 16(2), 159–176, 1992.

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
