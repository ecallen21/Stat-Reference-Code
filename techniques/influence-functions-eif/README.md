# Influence Functions + EIF (Reference §33.11)

**The influence function** of a statistical functional `T` at a
distribution `F` is its Gâteaux derivative in the direction of a point
mass:

```
IF(x; T, F) = lim_{t → 0} [ T((1 − t) F + t δ_x) − T(F) ] / t.
```

Hampel (1974) introduced the concept as a robustness diagnostic; the
**efficient influence function** (EIF) — Bickel-Klaassen-Ritov-Wellner
(1993) — is the object that defines semiparametric-efficient
estimators.

## Two headline uses

1. **Standard errors** via the sample influence:
   `SE(T̂) = sd(IF(X_i)) / √n`. For the sample mean this recovers the
   textbook `s / √n`.
2. **Robustness diagnostics**: `sup |IF|` is the gross-error sensitivity
   — infinite for the mean (unbounded IF) but finite for the median
   (bounded IF).

## Common IFs

| Functional     | Influence function                              |
|---------------|-------------------------------------------------|
| Mean          | `x − μ`                                         |
| Variance      | `(x − μ)² − σ²`                                 |
| Median        | `sign(x − m) / (2 f(m))`                        |
| Quantile τ    | `(τ − 𝟙[x ≤ q_τ]) / f(q_τ)`                     |
| E[Y] under R  | `μ̂(X) + R · (Y − μ̂(X)) / π̂(X)  −  θ`         |

## When to use

- **Building efficient causal / semiparametric estimators** (AIPW,
  TMLE, double ML) — the estimator has an EIF; the variance is
  `Var(EIF) / n`.
- **Robustness auditing** of a statistical procedure — bounded IF ⇒
  robust to outliers.
- **Sensitivity analysis** — trace how a specific observation moves
  the estimate.

## Files

- `python/influence_functions_eif.py` — from-scratch `if_mean`,
  `if_median`, `if_variance`, `se_from_if`. Demo confirms
  `IF-based SE = plug-in SE` for the mean (0.0589 both). Adds an
  outlier at `15.0`: mean shifts 0.050, median shifts 0.0008;
  `IF_mean(15) = 14.99` (unbounded), `IF_median(15) = 1.43`
  (bounded) — the numerical version of the robustness contrast.
- `r/influence_functions_eif.R` — `npcausal` / `drtmle` /
  `robustbase` (R); `econml` / `dowhy` (Python).

## Assumptions & caveats

- **Nonparametric density at the median** — for `IF_median` we need
  `f(m)`; the demo uses a Gaussian KDE at the sample median.
- **Sample IF ≠ theoretical IF** — the empirical version is the
  correct plug-in but noisy for small n.
- **EIF depends on the model class** — misspecifying the model
  space changes the EIF and hence the achievable efficiency.
- **Cross-fitting** — for estimators using ML nuisance models, split-
  sample cross-fitting is required to control the second-order term.
- **Higher-order influence** functions exist (Robins-Li-Tchetgen-van
  der Vaart 2008) and give smaller bias in some semiparametric problems.

## Related in this repo

- `semiparametric-efficiency` — the direct use of EIF for AIPW.
- `sandwich-robust-se` — the Huber-White SE is a plug-in IF variance.
- `tmle-doubly-robust` — targeted MLE uses the EIF as its estimating
  equation.
- `bca-bootstrap`, `jackknife` — nonparametric SE alternatives.
- `robust-regression`, `mm-estimators-robust` — bounded-IF estimators.

## Run

```
python techniques/influence-functions-eif/python/influence_functions_eif.py
Rscript techniques/influence-functions-eif/r/influence_functions_eif.R
```

**Refs:** Hampel, F. "The influence curve and its role in robust estimation." *JASA*, 1974; Bickel, P.J., Klaassen, C.A.J., Ritov, Y. & Wellner, J.A. *Efficient and Adaptive Estimation for Semiparametric Models*, Johns Hopkins University Press, 1993; van der Vaart, A.W. *Asymptotic Statistics*, Cambridge University Press, 2000 (Ch. 25).

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
