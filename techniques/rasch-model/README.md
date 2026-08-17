# Rasch Model / 1PL IRT (Reference §22.5)

Simplest item response theory (IRT) model for dichotomous (correct / incorrect) items (Rasch 1960).

```
Pr(y_ij = 1 | θ_i, b_j) = 1 / (1 + exp(−(θ_i − b_j)))
```

- `θ_i` — person ability
- `b_j` — item difficulty (location on the same latent scale as `θ`)
- **Common discrimination = 1** (the "one parameter" in 1PL)

## Special properties

- **Sufficient statistics** — person's raw score is sufficient for `θ` given `b`; item's raw score is sufficient for `b` given `θ`. This is what makes Rasch distinct from 2PL / 3PL.
- **Specific objectivity** — comparisons between persons don't depend on which items were used (and vice versa).
- **Additive on the logit scale** — item difficulty and person ability enter symmetrically.

## Estimation

- **Joint MLE (JML)** — maximize joint likelihood over `(θ, b)`. Simple; slight small-sample bias.
- **Conditional MLE (CML)** — condition on person raw scores to eliminate `θ`; consistent even in small samples. `eRm::RM`.
- **Marginal MLE (MML)** — integrate `θ` under a `Normal(0, σ²)` prior. Standard for research. `ltm::rasch` / `TAM::tam.mml`.

## Files

- `python/rasch_model.py` — from-scratch coordinate-ascent JML with centered difficulty for identifiability. Demo (n = 300, J = 20): correlation of estimated `b` with truth = 0.99; correlation of estimated `θ` with truth = 0.74.
- `r/rasch_model.R` — pointers to `eRm::RM`, `ltm::rasch`, `TAM::tam.mml`.

## When to use

- **Educational testing** with dichotomous items where you want a fair, additive scale.
- **Health outcome measurement** (PROMIS uses Rasch-family models extensively).
- **When strict measurement properties matter** — Rasch is the only IRT model with sufficient statistics.

## When to prefer 2PL / 3PL

- Item discriminations differ substantively (some items separate high vs low ability much better than others).
- Guessing matters (multiple-choice) — use 3PL.

## Assumptions & caveats

- **Unidimensional** trait `θ` — check with EFA or IRT fit statistics.
- **Local independence** — items are independent given `θ`. Violations distort fit.
- **Monotonicity** — Pr(correct) increases with `θ`.
- **Identifiability**: fix `sum(b) = 0` or `θ_1 = 0`.

## Run

```
python techniques/rasch-model/python/rasch_model.py
Rscript techniques/rasch-model/r/rasch_model.R
```

**Refs:** Rasch, G. *Probabilistic Models for Some Intelligence and Attainment Tests*, Danish Institute for Educational Research, 1960; Andrich, D. *Rasch Models for Measurement*, Sage, 1988; van der Linden, W.J. *Handbook of Item Response Theory*, CRC, 2016.

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
