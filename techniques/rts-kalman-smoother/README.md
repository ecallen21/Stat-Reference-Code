# RTS Kalman Smoother (Reference §47.127)

Rauch, Tung & Striebel (1965). Backward pass over the Kalman-
filtered posterior that returns the SMOOTHED posterior
`p(x_t | y_{1:T})`:

    C_t = P_t^f Fᵀ (P_{t+1}^p)⁻¹
    x_t^s = x_t^f + C_t (x_{t+1}^s − x_{t+1}^p)
    P_t^s = P_t^f + C_t (P_{t+1}^s − P_{t+1}^p) C_tᵀ.

Provides tighter posterior variances than filtering alone and
better estimates near the start of the series.

## Files

- `python/rts_kalman_smoother.py` — from-scratch KF + RTS
  backward pass. Demo (n=60 random-walk observations, Q=0.01,
  R=0.5):
  - Kalman filter RMSE = 0.333, mean posterior var = 0.077
  - **RTS smoother RMSE = 0.178, mean posterior var = 0.039**
  - Smoother tightens variance by 49 %.
- `r/rts_kalman_smoother.R` — `KFAS::KFS`, `dlm::dlmSmooth` (R);
  `filterpy.kalman.rts_smoother`, `statsmodels` (Python).

## When to use

- **Offline state estimation** — you have the full time series in
  hand.
- **Missing / outlier-corrupted observations** — smoother
  interpolates back through the record.
- **EM for HMM / state-space** — forward-backward = RTS in the
  Gaussian case.
- **Sensor fusion / trajectory reconstruction**.

## When NOT to use

- **Real-time / streaming** — filter (forward pass) only; smoother
  needs the future.
- **Highly nonlinear / non-Gaussian systems** — use particle
  smoother, extended-RTS, or unscented-RTS.
- **Very long series with memory constraints** — the RTS pass
  needs stored filter covariances.

## Assumptions & caveats

- **Linear-Gaussian** F, H, Q, R — else use EKF-RTS or UKS.
- **Numerical stability** — square-root form (Bierman 1977)
  avoids negative Ps.
- **Fixed-interval RTS** — variants: fixed-lag, fixed-point
  smoothers.
- **Two-pass memory** vs one-pass information smoothers.

## Related in this repo

- `state-space-kalman`, `state-space-models`,
  `extended-kalman-filter`, `unscented-kalman-filter`,
  `ensemble-kalman-filter`, `particle-filter-smc` — filter
  family.
- `dynamic-mode-decomposition-dmd`, `sindy-sparse-dynamics`,
  `neural-ode`, `hnn-hamiltonian-neural-networks` — data-driven
  dynamics siblings.
- `arima`, `sarima-arimax`, `exponential-smoothing`,
  `prophet-forecasting`, `hierarchical-forecasting` — TS
  forecasting cousins.

## Run

```
python techniques/rts-kalman-smoother/python/rts_kalman_smoother.py
Rscript techniques/rts-kalman-smoother/r/rts_kalman_smoother.R
```

**Refs:** Rauch, H.E., Tung, F. & Striebel, C.T. "Maximum likelihood estimates of linear dynamic systems." *AIAA Journal* 3(8): 1445-1450, 1965; Särkkä, S. *Bayesian Filtering and Smoothing.* Cambridge Univ Press, 2013.

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
