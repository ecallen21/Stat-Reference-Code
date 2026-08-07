# Truncated Regression (Reference §5.18)

**Truncated** sample: subjects whose `y` falls outside a threshold are **never observed** — not just recorded at the boundary as in Tobit. Enroll only households earning below the poverty line, test only students above a cutoff, price only quotes accepted below a cap. Naive OLS on the truncated sample is biased **more** severely than the Tobit-censored case.

## Truncated-normal MLE

```
y_i^* = X_i β + ε_i,       ε_i ~ N(0, σ²)
observed only if L < y_i^* < U
density:  f(y | X) = φ((y − Xβ)/σ) / σ
                    / [Φ((U − Xβ)/σ) − Φ((L − Xβ)/σ)]
```

The denominator corrects for the **selection** into the observed sample.

## Truncation vs censoring

|                | Censored (Tobit)                       | Truncated                                |
|----------------|----------------------------------------|------------------------------------------|
| Boundary obs   | recorded at limit                      | not in the dataset                       |
| Likelihood     | mixture of density + boundary mass     | density divided by truncation prob       |
| Bias if OLS    | severe                                 | more severe                              |

## Files

- `python/truncated_regression.py` — from-scratch BFGS MLE for left / right / two-sided truncation. Demo: upper-truncated at 3 (n = 500 accepted from 200-at-a-time rejection sampling): OLS gives β = (0.75, 1.71), truncated MLE gives β = (0.94, 1.92), true β = (1.0, 2.0); σ̂ = 0.99.
- `r/truncated_regression.R` — `truncreg::truncreg(y ~ x, point = 3, direction = "right")`.

## When to use

- **Sampling frame** excludes subjects outside a threshold (unemployment surveys of the unemployed, hospital records for admitted patients).
- **Rejection-sampled** experimental data where selection depends on the outcome.

## When to use Tobit instead

- Boundary observations are **present in the data**, just capped at the boundary — Tobit.
- The unobserved-outside-range structure is truncation.

## Assumptions & caveats

- **Normal errors** — MLE is sensitive to misspecification. Consider more flexible truncated families (e.g. truncated Student-t) for heavy tails.
- **Known truncation points** — the model must know `L` and/or `U`; unknown truncation is much harder.
- **Sample size** — MLE loses efficiency compared to un-truncated OLS in proportion to the truncation probability.

## Run

```
python techniques/truncated-regression/python/truncated_regression.py
Rscript techniques/truncated-regression/r/truncated_regression.R
```

**Refs:** Hausman, J.A. & Wise, D.A. "Social experimentation, truncated distributions, and efficient estimation." *Econometrica* 45(4), 919–938, 1977; Maddala, G.S. *Limited-Dependent and Qualitative Variables in Econometrics*, Cambridge, 1983.

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
