# Granger Causality (Reference §13.50)

Does past `X` help predict `Y` beyond what past `Y` already tells us?

```
Restricted:    y_t = c + Σ φ_k y_{t-k}                   + ε_t
Unrestricted:  y_t = c + Σ φ_k y_{t-k} + Σ ψ_k x_{t-k}    + ε_t

F  =  ((SSR_R − SSR_U) / p_x)  /  (SSR_U / (n − p_y − p_x − 1))
```

Small p → adding X's lags reduces SSR significantly → **X Granger-causes Y**.

## What "Granger cause" really means

**A prediction statement, not a causal one.** If X and Y are both driven by a common upstream Z that reaches X first, X will "Granger-cause" Y without causing it in any real sense. The name is unfortunate.

Guidelines:

- **Test both directions.** Report `X → Y` and `Y → X` side-by-side.
- **Rule out common causes** before making causal claims (either through prior knowledge or by adding suspected confounders to the VAR).
- Prefer the phrase *"X Granger-causes Y"* explicitly — never just "X causes Y."

## Multivariate extension

For K series, fit a VAR (see [`var-cointegration`](../var-cointegration)) and test whether the block of coefficients on X's lags in the Y equation is jointly zero. Same F-test, larger design matrix.

## Files

- `python/granger_causality.py` — from-scratch nested-regression F-test + bidirectional helper. On a DGP where X truly causes Y, F=42 (p≈3e-36); reverse direction F=0.48 (p=0.79). Matches statsmodels `grangercausalitytests` exactly.
- `r/granger_causality.R` — thin wrapper around `lmtest::grangertest`.

## Assumptions

- Stationary series (difference first if not).
- Correct lag length — too few misses causality; too many wastes df.
- Linear predictive relationships. Nonlinear "cause" needs kernel Granger or transfer-entropy methods.

## Run

```
python techniques/granger-causality/python/granger_causality.py
Rscript techniques/granger-causality/r/granger_causality.R
```

**Refs:** Granger, C.W.J. "Investigating causal relations by econometric models and cross-spectral methods." *Econometrica* 37(3), 424–438, 1969; Pearl, J. *Causality*, 2nd ed., Cambridge UP, 2009 (Ch. 1 on why prediction ≠ causation); Lütkepohl, H. *New Introduction to Multiple Time Series Analysis*, Springer, 2005 (Ch. 2.3).

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
