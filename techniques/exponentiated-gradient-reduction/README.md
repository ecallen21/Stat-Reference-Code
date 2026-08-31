# Exponentiated-Gradient Reduction (Reference Ch 31 Fairness)

**Reduce fair classification to a sequence of cost-sensitive classification
problems.** Agarwal, Beygelzimer, Dudík, Langford & Wallach (2018) — the
theoretical backbone of `fairlearn.reductions`.

## Idea

Constrained ERM

```
min_h  err(h)   s.t.   |E[h(X) | A = a] − E[h(X)]| ≤ ε   for every group a.
```

Lagrangian:

```
L(h, λ) = err(h) + Σ_a λ_a^+ (E[h|A = a] − E[h] − ε)
                 + Σ_a λ_a^− (E[h] − E[h|A = a] − ε)
```

Saddle-point solved by **exponentiated gradient** on `λ` + **best
response** by any base learner:

```
Repeat T rounds:
  (1) Convert current λ to per-example weights (or costs).
  (2) Refit the base classifier with those weights.
  (3) Measure per-group violation; multiplicatively update λ.
Return the UNIFORM MIXTURE of the T classifiers (randomised meta-classifier).
```

The `ε` knob is a hard fairness tolerance; the reduction returns a
**Pareto-optimal** classifier subject to that tolerance.

## Cost-sensitive weights (DP variant)

For over-selected groups the reduction *up-weights* their negative
examples (encouraging the classifier to say 0 more often there); for
under-selected groups it *up-weights* their positive examples.

## When to use

- **You want a formal fairness tolerance** rather than a soft
  regulariser.
- **You can wrap any base learner** — decision tree, logistic, GBM,
  neural net — through a `sample_weight` API.
- **Supports many fairness definitions** — demographic parity,
  equalised odds, equal opportunity, bounded group loss.

## When NOT to use

- **You cannot afford multiple retrains** — each EG round refits from
  scratch (or warm-starts).
- **Deterministic decisions required** — the output is a *randomised*
  classifier; deterministic use requires majority voting or picking a
  single round.

## Files

- `python/exponentiated_gradient_reduction.py` — from-scratch **DP
  reduction** with two-sided multipliers, cost-sensitive weights, and
  a uniform-mixture predictor. Base learner: weighted logistic
  regression. Demo: **ERM DP ratio 0.415 → EG(T = 30) DP ratio 0.737**;
  accuracy 0.849 → 0.805.
- `r/exponentiated_gradient_reduction.R` — `fairml` /
  `mlr3fairness`; `fairlearn.reductions.ExponentiatedGradient` in
  Python.

## Assumptions & caveats

- **η (learning rate on λ) and T** matter — too small ⇒ no effect;
  too large ⇒ oscillation. The demo uses `η = 5`, `T = 30`.
- **Cost sign matters** — the correct direction requires cost-
  sensitive labels; a plain BCE weight with the wrong sign can drive
  the DP ratio the *wrong* way. See implementation comments.
- **Randomised meta-classifier** — the output is a distribution over
  the T base classifiers; for deployment use majority voting or the
  single best-ε classifier.
- **Multi-attribute fairness** — the reduction generalises but the
  simplex over `λ` becomes higher-dimensional; convergence slows.
- **Constraint definition** — equalised odds needs constraints
  conditional on `Y`; the demo above handles only DP.

## Related in this repo

- `demographic-parity`, `equalized-odds` — the criteria this reduction
  enforces.
- `reweighing-preprocessing`, `adversarial-debiasing`,
  `equalized-odds-postprocessing`, `fair-representations-lfr` —
  sibling mitigations.
- `distributionally-robust-optimization` — a related worst-group-risk
  reduction.
- `logistic-regression`, `random-forest` — the base learners this
  reduction wraps.

## Run

```
python techniques/exponentiated-gradient-reduction/python/exponentiated_gradient_reduction.py
Rscript techniques/exponentiated-gradient-reduction/r/exponentiated_gradient_reduction.R
```

**Refs:** Agarwal, A. et al. "A reductions approach to fair classification." *ICML*, 2018; Freund, Y. & Schapire, R.E. "A decision-theoretic generalisation of on-line learning and an application to boosting (AdaBoost / EG)." *JCSS*, 1997.

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
