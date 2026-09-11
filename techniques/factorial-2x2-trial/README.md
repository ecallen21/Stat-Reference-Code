# 2x2 Factorial Trial (Reference §47.267)

Piantadosi (2005); McAlister et al (2003). Randomise TWO
interventions simultaneously in a 2x2 grid:

```
outcome = μ + a A + b B + c (A · B) + e
```

Analysis estimates the main effect of A, the main effect of
B, and the A·B interaction. Efficient — two interventions
with (almost) the sample size of a single-arm trial. BUT the
presence of an interaction inflates variance and can mask
main effects.

## Files

- `python/factorial_2x2_trial.py` — OLS fit + t-tests with
  and without interaction. Demo: n=100/cell (N=400). No
  interaction (c=0): recovers a=0.50, b=0.40 with SE ~0.10.
  With interaction (c=0.6): all three effects a=0.49, b=0.39,
  a·b=0.53 recovered with p<0.01. Factorial N=800 gives SE(A)
  ~0.07 vs single-trial N=400 gives SE(A) ~0.10 — factorial
  gains info on B "for free" when no interaction.
- `r/factorial_2x2_trial.R` — `base::lm + anova`,
  `afex::aov_ez`, `emmeans`, `pwr::pwr.f2.test` (R);
  `statsmodels.formula.api.ols('y ~ A*B')`, from-scratch
  (Python).

## When to use

- **Testing two interventions** where you want to answer both
  research questions in one trial.
- **Sample size limitations** — factorial "doubles up" the N
  for main-effect tests when interaction is small.
- **Biological / mechanistic interest** in whether A and B
  interact.

## When NOT to use

- **Large expected interaction** — main-effect interpretation
  becomes conditional and confusing; run two separate trials.
- **One arm dominates enrolment barriers** — factorial forces
  all four cells and can create logistical strain.
- **Adaptive enrichment desired** — factorial is a fixed
  design; response-adaptive is more flexible.

## Assumptions & caveats

- **Interaction detection under-powered** — the A·B test
  typically needs 4× the sample of a main-effect test at the
  same magnitude.
- **Marginal-effect interpretation** requires "half A, half
  not-A" logic — clearly document in publication.
- **Balanced cells** simplify analysis; unbalanced factorial
  needs Type II / Type III SS choice.
- **Composite endpoint risk** — if either intervention hits a
  ceiling / floor, additive linear model may misfit.

## Related in this repo

- `cluster-randomized-trial`, `stepped-wedge-design` — other
  cluster / adaptive designs.
- `fractional-factorial`, `split-plot-design` — DOE relatives.
- `manova`, `permanova` — multivariate outcome analogues.

## Run

```
python techniques/factorial-2x2-trial/python/factorial_2x2_trial.py
Rscript techniques/factorial-2x2-trial/r/factorial_2x2_trial.R
```

**Refs:** Piantadosi, S. *Clinical Trials: A Methodologic Perspective*, 2nd ed., Wiley, 2005; McAlister, F.A. et al. "Analysis and reporting of factorial trials: A systematic review." *JAMA*, 289(19): 2545-2553, 2003.

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
