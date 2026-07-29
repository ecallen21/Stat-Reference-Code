# State-Space Models + Kalman Filter (Reference §13.17; also covers §13.20, §13.55)

General linear-Gaussian state-space model:

```
x_t  =  F · x_{t-1}  +  w_t          w_t ~ N(0, Q)     (state)
y_t  =  H · x_t      +  v_t          v_t ~ N(0, R)     (observation)
```

Extremely flexible — encompasses ARIMA, exponential smoothing, structural TS, dynamic regression, and many others as special cases.

## Kalman Filter — recursive Bayesian update

```
Predict:  x_pred_t = F x_{t-1|t-1}
          P_pred_t = F P_{t-1|t-1} F' + Q
Update:   K_t      = P_pred_t H' (H P_pred_t H' + R)⁻¹
          x_{t|t}  = x_pred_t + K_t (y_t − H x_pred_t)
          P_{t|t}  = (I − K_t H) P_pred_t
```

**Innovation** `y_t − H x_pred_t` and its variance `S` accumulate the log-likelihood, so state-space models can be fit by likelihood MLE just like ARIMA.

## Two canonical models

- **Local level** — state is the current mean; `x_t = x_{t-1} + w`. Adaptive average that reacts to level changes.
- **Local linear trend** — state is `(level, slope)`; both drift stochastically. Reduces to Holt smoothing when noise is estimated by MLE.

## §13.20 DLMs / §13.55 UCM

Same mathematical machinery, different naming:

- **DLM** (Dynamic Linear Models) — West & Harrison Bayesian framework.
- **UCM** (Unobserved Components Model) — Harvey's structural time-series formulation with named components (level / slope / seasonal / cycle / irregular).

## Files

- `python/state_space_kalman.py` — generic Kalman filter + local-level and local-linear-trend wrappers + Kalman forecast with 95% intervals. On synthetic data, filtered state tracks true state with RMSE ~0.65.
- `r/state_space_kalman.R` — base `stats::StructTS` + `KFAS::SSModel`.

## Assumptions

- Linear dynamics and observation.
- Gaussian noise (Kalman filter is BLUE — best linear unbiased estimator — even without Gaussianity, but the log-likelihood assumes it).
- Time-invariant `F, H, Q, R` (this file); time-varying versions are straightforward extensions.
- Extended Kalman / Unscented Kalman handle nonlinear dynamics (not shipped).

## Run

```
python techniques/state-space-kalman/python/state_space_kalman.py
Rscript techniques/state-space-kalman/r/state_space_kalman.R
```

**Refs:** Kalman, R.E. "A new approach to linear filtering and prediction problems." *J. Basic Eng.* 82(1), 35–45, 1960; Harvey, A.C. *Forecasting, Structural Time Series Models and the Kalman Filter*, Cambridge UP, 1989; Durbin, J. & Koopman, S.J. *Time Series Analysis by State Space Methods*, 2nd ed., Oxford UP, 2012; Petris, G., Petrone, S. & Campagnoli, P. *Dynamic Linear Models with R*, Springer, 2009.

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
