# Fine-Gray Subdistribution Hazards (Reference §11.9)

Competing-risks alternative to the cause-specific Cox model. Fine & Gray (1999) parameterize a hazard on the **cumulative incidence function** (CIF) directly:

```
h_1^sub(t | X) = h_10^sub(t) · exp(Xβ)
```

where the **subdistribution hazard** for cause 1 is

```
h_1^sub(t) = lim Pr(t ≤ T ≤ t+dt, cause = 1 | T > t OR (T ≤ t AND cause ≠ 1))
```

Subjects who experience a competing event stay in the risk set with a **time-decaying weight** (from IPCW of the censoring distribution) so that a covariate that raises `h_1^sub` directly raises `CIF_1`.

## Cause-specific vs Fine-Gray

|                       | Cause-specific Cox                     | Fine-Gray                              |
|-----------------------|----------------------------------------|----------------------------------------|
| Target hazard         | rate among truly at-risk               | rate on CIF scale                      |
| Interpretation of β   | effect on **hazard rate**              | effect on **cumulative incidence**     |
| Competing subjects    | removed from risk set (censored)       | kept with decaying weight              |
| Best for              | etiology (why does event happen?)      | prediction (probability of event)      |

They can give **different signs**: a covariate that increases both the cause-of-interest hazard and the competing-event hazard can produce a positive cause-specific β but a null Fine-Gray β.

## Estimation

Weighted partial-likelihood analog of Cox. Weight subject `i` at time `t`:

```
w_i(t) = G(t) / G(min(T_i, t))       if T_i > t OR (T_i ≤ t AND cause ≠ target)
```

where `G(t) = P(C > t)` is the KM estimator of the censoring distribution.

## Files

- `python/fine_gray.py` — from-scratch weighted partial-likelihood optimization with IPCW censoring weights. Demo (n = 400, cause-specific β = 0.6 on cause 1 and β = −0.2 on competing): subdistribution HR = 1.60 (β̂ = 0.47), smaller than the cause-specific 0.6 because competing risk decreases with `x` — increasing `x` makes cause 1 happen sooner AND competing event less likely, both of which boost CIF_1.
- `r/fine_gray.R` — `cmprsk::crr` (Gerds-Scheike; the canonical R implementation).

## When to use

- **Predicting the probability** that a subject develops cause 1 by time `t`.
- **Clinical decision-making** where CIF is the relevant quantity ("what is the 5-year risk of relapse?").
- **Regulatory settings** where absolute risk (CIF) is the endpoint.

## When to prefer cause-specific Cox

- **Etiologic inference** — asking whether `X` biologically causes cause 1.
- **Rare competing events** — cause-specific and Fine-Gray coincide when competing risk is negligible.
- **Reporting alongside** Fine-Gray to give a complete picture.

## Assumptions & caveats

- **Proportional subdistribution hazards** — check with residuals as in Cox.
- **IPCW weights** rely on the censoring distribution being estimable — problematic under informative censoring.
- **Report both** cause-specific and Fine-Gray estimates when the story matters.

## Run

```
python techniques/fine-gray/python/fine_gray.py
Rscript techniques/fine-gray/r/fine_gray.R
```

**Refs:** Fine, J.P. & Gray, R.J. "A proportional hazards model for the subdistribution of a competing risk." *JASA* 94(446), 496–509, 1999; Andersen, P.K., Geskus, R.B., de Witte, T. & Putter, H. "Competing risks in epidemiology: possibilities and pitfalls." *Int. J. Epidemiol.* 41(3), 861–870, 2012.

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
