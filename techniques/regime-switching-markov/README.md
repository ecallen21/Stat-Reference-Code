# Markov-Switching Model (Reference §13.14, §13.15)

Hidden Markov state `S_t ∈ {1, ..., K}` drives the parameters of a time series:

```
y_t | S_t = k    ~ N(μ_k, σ_k²)              (Gaussian emissions; can extend to AR(p))
Pr(S_t = j | S_{t−1} = i) = P[i, j]           (Markov transition matrix)
```

Introduced by Hamilton (1989) for macro business-cycle modeling. Common uses:

- **Macro**: expansion vs recession regimes with distinct mean growth.
- **Finance**: low-vol vs high-vol regimes (Markov-switching GARCH / SWARCH, Hamilton & Susmel 1994).
- **Ecology / medicine**: baseline vs disease-onset regime for biomarkers.

## Related model families

- **Threshold AR (TAR / SETAR)** — regimes triggered by a threshold on an observed variable, not a hidden state.
- **Smooth-Transition AR (STAR)** — regimes blend with a logistic weight.
- **Markov-switching GARCH** — regime-dependent volatility dynamics.

## Estimation (EM / Baum-Welch)

- **E-step**: forward-backward gives `γ_t(k) = Pr(S_t = k | y_{1:T})` and pairwise `ξ_t(i, j)`.
- **M-step**: reweighted mean, variance, and transition-matrix updates.

## Files

- `python/regime_switching_markov.py` — from-scratch forward-backward EM in log-space with Gaussian emissions; sorted regimes by mean for identifiability. Demo recovers 96.8% classification accuracy on a 2-regime series; matches `statsmodels.tsa.regime_switching.MarkovRegression` on regime variances (0.527, 2.411).
- `r/regime_switching_markov.R` — `depmixS4::depmix` (Gaussian HMM) or `MSwM::msmFit`.

## When to use

- Time series with visibly different behavior in different periods that isn't cleanly triggered by an observed threshold.
- Modeling regime **persistence** — the Markov transition captures how sticky regimes are (`P[i, i]` near 1 → long spells).
- When you need **smoothed regime probabilities** (posterior probability that each observation belongs to each regime) rather than a hard clustering.

## Assumptions & caveats

- Fixed number of regimes `K` — pick with AIC / BIC or by cross-validation.
- Label switching is inherent — sort regimes by a monotone parameter (mean, variance) after fitting.
- Local optima: use multiple random starts; EM converges to the local optimum of the log-likelihood.
- Convergence to the boundary (`σ_k → 0`) if a regime picks up too few points; add a small ridge.

## Run

```
python techniques/regime-switching-markov/python/regime_switching_markov.py
Rscript techniques/regime-switching-markov/r/regime_switching_markov.R
```

**Refs:** Hamilton, J.D. "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica* 57(2), 357–384, 1989; Hamilton, J.D. & Susmel, R. "Autoregressive conditional heteroskedasticity and changes in regime." *J. Econometrics* 64(1–2), 307–333, 1994.

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
