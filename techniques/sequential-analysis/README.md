# Sequential Analysis / SPRT (Reference §37.6)

Wald (1945). At each observation, compute the **log-likelihood
ratio**

```
Λ_n  =  Σ_{i=1..n} log( f_1(x_i) / f_0(x_i) )
```

and decide:

```
Λ_n ≥ B         → reject H_0.
Λ_n ≤ A         → accept H_0.
A < Λ_n < B     → continue sampling.

A = log(β / (1 − α)),   B = log((1 − β) / α).
```

## Efficiency

For the same `(α, β)`, SPRT's **expected sample size** is dramatically
smaller than a fixed-n test — the sample-size guarantee is what made
sequential testing famous in WWII quality control.

## When to use

- **Adaptive clinical trials** (with early stopping).
- **Streaming / online monitoring** with a sharp hypothesis pair.
- **Quality inspection** where sampling costs matter.

## When NOT to use

- **Composite hypotheses** — SPRT is for two simple hypotheses;
  use group-sequential designs (O'Brien-Fleming, Pocock) for composite.
- **Effect sizes far from your alternative** — SPRT can take arbitrarily
  long to decide.

## Files

- `python/sequential_analysis.py` — from-scratch Bernoulli SPRT for
  H_0: p = 0.5 vs H_1: p = 0.7 with α = β = 0.05. 500 trials:
  - When H_1 is true: mean stopping = 34, 478/500 correctly reject.
  - When H_0 is true: mean stopping = 33, 474/500 correctly accept.
  - **Fixed-n test for the same errors requires ~63 obs**;
    SPRT averages 34 (≈ 2× efficiency).
- `r/sequential_analysis.R` — `gsDesign`, `rpact`, `ldbounds`,
  `sprtt` (R); `scipy` (Python).

## Assumptions & caveats

- **Simple hypotheses** — extension to composite requires
  generalisations (invariance, mixture SPRT).
- **Terminating** — SPRT has probability 1 of stopping under either
  hypothesis but can be very slow when true parameter is between H_0
  and H_1.
- **Overshoot** — actual (α, β) can differ slightly from nominal
  because Λ can jump past a boundary.
- **Group-sequential extensions** — Pocock, O'Brien-Fleming with
  alpha-spending for pre-planned interim analyses.

## Related in this repo

- `cusum-charts` — likelihood-ratio interpretation.
- `bayesian-ab-testing`, `multi-armed-bandits` — Bayesian sequential
  alternatives.
- `canary-deployment`, `shadow-deployment` — MLOps sequential
  monitoring.
- `power-analysis` (if present) — fixed-n analogue.

## Run

```
python techniques/sequential-analysis/python/sequential_analysis.py
Rscript techniques/sequential-analysis/r/sequential_analysis.R
```

**Refs:** Wald, A. "Sequential tests of statistical hypotheses." *Annals of Mathematical Statistics*, 1945; Whitehead, J. *The Design and Analysis of Sequential Clinical Trials*, Wiley, 1997.

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
