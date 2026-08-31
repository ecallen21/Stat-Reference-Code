# Reweighing Pre-processing (Reference Ch 31 Fairness)

Assign per-example training weights so that the protected attribute `A`
and label `Y` become **statistically independent** in the weighted
sample. Kamiran & Calders (2012) — the *original* fairness pre-processor,
cheap and model-agnostic.

## Formula

For each cell `(A = a, Y = y)`:

```
w_{a, y}  =  ( P(A = a) · P(Y = y) )  /  P(A = a, Y = y)
```

- Over-represented cells (e.g. majority × favoured label) get weight
  `< 1`.
- Under-represented cells (minority × favoured label) get weight `> 1`.
- Weighted `(A, Y)` product distribution equals the marginal product
  by construction.

## How to use

1. Compute `w_{a, y}` on the training set only.
2. Train any weighted-loss classifier (logistic, GBM, neural net) with
   sample weights `w_i = w_{A_i, Y_i}`.
3. Deploy the classifier on unseen data — no fairness-specific step
   needed at inference.

## When to use

- **Any binary-label + binary-attribute setting** — one line to
  compute; drops into any learner with `sample_weight` support.
- **Pre-processing preferred** — you cannot modify the training
  pipeline downstream.
- **First fairness intervention to try** — usually improves DP with
  small accuracy cost.

## When NOT to use

- **Multiple protected attributes** — cell counts explode; use
  `adversarial-debiasing` or `exponentiated-gradient-reduction`.
- **Very small groups** — some weights get huge (Effective Sample Size
  collapses); regularise.
- **Non-DP targets** — for equalized odds or predictive parity, use a
  post-processor targeted at that criterion.

## Files

- `python/reweighing_preprocessing.py` — from-scratch
  `kamiran_calders_weights` + weighted logistic regression on
  synthetic data (group-0 base rate 0.60, group-1 base rate 0.25 +
  a proxy feature that correlates with group). Result: **DP ratio
  0.42 → 0.59** at an accuracy cost of 0.814 → 0.802.
- `r/reweighing_preprocessing.R` — `fairml` / `fairmodels` or plain
  `glm(..., weights = w)`; `aif360.algorithms.preprocessing
  .Reweighing` in Python.

## Assumptions & caveats

- **Only debases (A, Y) correlation** — the classifier may still infer
  A from other features (proxy variables), keeping the demographic
  disparity.
- **Effective sample size** drops when weights vary a lot; regularise
  the classifier or clip weights.
- **Downstream calibration** may worsen — retest calibration after
  reweighing.
- **Multi-valued Y or A** — the formula generalises to `w = P(A)·P(Y)
  / P(A, Y)` per cell; watch small counts.
- **Target metric is DP-family** — reweighing does NOT enforce
  equalized odds; use dedicated methods for that.

## Related in this repo

- `demographic-parity`, `disparate-impact` — the criteria reweighing
  targets.
- `adversarial-debiasing`, `exponentiated-gradient-reduction` — in-
  training alternatives.
- `equalized-odds-postprocessing` — post-hoc alternative.
- `class-imbalance` — the same weighting trick applied to `Y` only.
- `covariate-shift-adaptation` — a superset (importance-weighting for
  arbitrary shifts).

## Run

```
python techniques/reweighing-preprocessing/python/reweighing_preprocessing.py
Rscript techniques/reweighing-preprocessing/r/reweighing_preprocessing.R
```

**Refs:** Kamiran, F. & Calders, T. "Data preprocessing techniques for classification without discrimination." *Knowledge and Information Systems*, 2012; Kamiran, F. & Žliobaitė, I. "Explainable and non-explainable discrimination in classification." In *Discrimination and Privacy in the Information Society*, Springer, 2013.

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
