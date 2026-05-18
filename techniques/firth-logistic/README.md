# Firth's Penalized Logistic Regression (Reference §7.7, §7.51)

The fix for **separation** in logistic regression.

## The problem: separation

If a predictor (or combination) perfectly classifies the outcome — e.g. "every patient with biomarker > 5 had the event, every patient with biomarker ≤ 5 didn't" — then the standard ML estimator **diverges**:

- `β̂ → ±∞`
- `SE(β̂) → ∞`
- Convergence warnings or huge coefficients on the output

The demo here constructs separation deliberately; the standard ML logistic gives `β̂_x₁ ≈ 54` and SE that's nonsense.

## The fix: Firth (1993)

Maximize a **penalized** log-likelihood:

`L*(β) = L(β) + ½ · log|I(β)|`

where `I(β)` is the Fisher information. The `log|I(β)|` term is the **Jeffreys prior** in Bayesian language; it shrinks β away from the boundary by an `O(1/n)` correction.

The MLE-like properties (consistency, asymptotic normality) are preserved; the bias correction is a strict improvement in small samples even when there's no separation.

## Algorithm — modified IRLS

Standard logistic IRLS uses `wᵢ = πᵢ(1 − πᵢ)` and working response `zᵢ = ηᵢ + (yᵢ − πᵢ)/wᵢ`.

Firth's modification: add `hᵢ · (½ − πᵢ)` to the numerator of the working response, where `hᵢ = diag(X (X'WX)⁻¹ X' W)` is the hat-matrix diagonal. Iterate to convergence.

## When to use it

- Small `n` with a strong predictor.
- Rare-outcome studies where some covariate level has 0 events.
- "Quasi-separation" — almost-perfect classification.
- General defensive choice in clinical regression when standard MLE wobbles.

For very-rare-event prediction at scale, **ridge** / **lasso logistic** (`techniques/regularization` applied to logistic loss) are alternatives.

## Files
- `python/firth_logistic.py` — from-scratch Firth IRLS via the modified working response; compares to ML logistic on a deliberately separated example.
- `r/firth_logistic.R` — from-scratch + `logistf::logistf` (the canonical R implementation).
- PySpark: N/A — Firth is small-sample-focused.

## Run
```
python techniques/firth-logistic/python/firth_logistic.py
Rscript techniques/firth-logistic/r/firth_logistic.R
```

**Refs:** Firth, "Bias Reduction of Maximum Likelihood Estimates," *Biometrika* 80(1), 27–38, 1993; Heinze & Schemper, "A Solution to the Problem of Separation in Logistic Regression," *Statistics in Medicine* 21, 2409–2419, 2002.

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
