# Multilevel 1-1-1 Mediation (Reference §12.22)

`X` (predictor), `M` (mediator), `Y` (outcome) are all **repeated measurements** per subject. Standard mediation with the "1-1-1" label = all three variables at level 1 (observation), clustered within subjects.

## The confound multilevel mediation fixes

Naive per-observation mediation confounds **between-subject** and **within-subject** effects. If some subjects always have high X, high M, high Y, but on any given day X and M and Y also fluctuate together within-person, a single per-observation fit averages the two. Which of the two you actually want depends on your substantive question.

## Bauer–Preacher–Gil (2006) decomposition

Person-mean-center X and M, then regress:

```
M   =  a_w · X_within  +  a_b · X_between  +  ε_M
Y   =  b_w · M_within  +  b_b · M_between
    +  c'_w · X_within +  c'_b · X_between  +  ε_Y
```

Two indirect effects:

```
indirect_within   =  a_w · b_w      ("day-to-day X shifts M, which shifts Y")
indirect_between  =  a_b · b_b      ("high-X subjects tend to have high M, which associates with high Y")
```

## CI via Monte Carlo

Draw many `(a, b)` from their joint sampling distribution (normal with SE from the OLS fit), compute the empirical percentile CI of `a · b`. More reliable than Sobel's product-SE, especially at small n.

## Files

- `python/multilevel_mediation.py` — person-mean-centered OLS approximation + Monte Carlo CI for both within and between indirect effects. Recovers `indirect_within = 0.20` (true 0.20) on the demo.
- `r/multilevel_mediation.R` — sketch pointing to `lme4` + `RMediation::medci` for random-slope proper multilevel fit; `bmlm` for Bayesian version.

## Assumptions

- **X causes M causes Y** — the causal ordering is a substantive claim; mediation math cannot verify it.
- Person means and person deviations are treated as separate predictors (essential to avoid confounding).
- Random-slope fits are more principled but more complex — see the R pointer.

## Run

```
python techniques/multilevel-mediation/python/multilevel_mediation.py
Rscript techniques/multilevel-mediation/r/multilevel_mediation.R
```

**Refs:** Bauer, D.J., Preacher, K.J. & Gil, K.M. "Conceptualizing and testing random indirect effects and moderated mediation in multilevel models." *Psych. Methods* 11(2), 142–163, 2006; Preacher, K.J. "Advances in mediation analysis." *Ann. Rev. Psych.* 66, 825–852, 2015; Zhang, Z., Zyphur, M.J. & Preacher, K.J. "Testing multilevel mediation using hierarchical linear models." *Org. Res. Methods* 12(4), 695–719, 2009.

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
