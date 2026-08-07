# Latin Square Design (Reference §16.6)

Experimental design that **blocks two nuisance factors** simultaneously with far fewer runs than a full three-factor factorial.

- **Row block**: `i = 1, ..., k` (e.g. day of experiment).
- **Column block**: `j = 1, ..., k` (e.g. lab technician / apparatus).
- **Treatment**: `t = ℓ(i, j)` — each treatment appears exactly once in every row and every column.

Example k = 3 Latin square:

```
A B C
B C A
C A B
```

**k² runs total** (vs `k³` for a full factorial), at the cost of assuming **no interactions** among the three factors.

## ANOVA model

```
y_ijt = μ + ρ_i + γ_j + τ_t + ε_ijt
SS_total  = SS_row + SS_col + SS_treatment + SS_error
df_total  = k² − 1
df_row    = df_col = df_treatment = k − 1
df_error  = (k − 1)(k − 2)
```

Need `k ≥ 3` for any error df.

## Extensions

- **Graeco-Latin square** — superimpose two orthogonal Latin squares to block a fourth factor.
- **Replicated Latin square** — pool multiple squares to gain error df.
- **Youden square** — incomplete Latin square (rectangular).

## Files

- `python/latin_square_design.py` — cyclic + randomized Latin-square generators + Latin-square ANOVA sums-of-squares. Demo (k = 5, true treatment effects 0, 0.5, 1.0, 1.5, 2.0): F_treatment = 37.6, p < 0.001; recovers row and column block variance.
- `r/latin_square_design.R` — base `aov(y ~ row + col + trt)` on a Latin-square layout.

## When to use

- **Two obvious nuisance factors** you want to block out (day × operator, plot × row, patient × visit-order).
- **Small experiments** where a full factorial is prohibitively expensive.
- **Cross-over trials** — Latin square with subjects × periods × treatments.

## Assumptions & caveats

- **No interactions** among row, column, and treatment — main-effects-only model. If interactions exist they get pooled into error and inflate the residual variance.
- **k ≥ 3** for at least one residual df; larger `k` gives more power.
- **Randomize** the assignment: pick a random Latin square from the population; don't reuse the same standard square across replications.
- **Balance** requires exactly one observation per (row, col) — no missing cells.

## Run

```
python techniques/latin-square-design/python/latin_square_design.py
Rscript techniques/latin-square-design/r/latin_square_design.R
```

**Refs:** Fisher, R.A. *The Design of Experiments*, Oliver & Boyd, 1935; Montgomery, D.C. *Design and Analysis of Experiments*, 9th ed., Wiley, 2017.

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
