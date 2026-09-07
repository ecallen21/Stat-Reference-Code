# Deep Survival Networks (Reference §11.27)

Katzman et al. (2018, DeepSurv). Neural-network extension of Cox PH:

    h(t | x) = h_0(t) · exp( g_θ(x) )

with `g_θ` any differentiable function. Loss = **negative Cox partial
likelihood**:

    L(θ) = −Σ_{i: e_i=1} [ g_θ(x_i) − log Σ_{j ∈ R(t_i)} exp(g_θ(x_j)) ]

## Related architectures

| Model | Idea |
|---|---|
| **DeepSurv** (Katzman 2018) | NN log-hazard, Cox loss |
| **DeepHit** (Lee 2018) | Discrete-time output head, competing risks |
| **Nnet-survival** | Discrete-time hazard head |
| **Coxtime** (Kvamme 2019) | Time-dependent NN log-hazard |
| **Neural ODE Survival** | Continuous-time via neural ODE |

## Files

- `python/deep_survival_network.py` — compact 1-hidden-layer
  DeepSurv from scratch with correct suffix-cumsum Cox partial-
  likelihood gradient. Demo (n=500, p=4, nonlinear true log
  hazard `0.5·x₀ + 0.7·x₁² − 0.4·x₂·x₃`): linear Cox C=0.622,
  DeepSurv (H=8) C=0.739 — the NN captures the nonlinearities.
- `r/deep_survival_network.R` — `survivalmodels` (DeepSurv +
  DeepHit + Coxtime + PC-Hazard wrappers), `torch` for R,
  `mlr3proba` (R); `pycox`, `auton-survival`, `torchsurv`,
  `sksurv` (Python).

## When to use

- **Nonlinear / interaction-heavy hazards** — Cox misses them; the
  NN captures without a specified functional form.
- **High-dimensional inputs** — images, embeddings, sequences.
- **Personalised treatment recommendation** — Katzman's original
  motivation.
- **Competing risks** — use DeepHit.

## When NOT to use

- **You need interpretable coefficients** — DeepSurv is a black-box
  risk score; use Cox or fractional-polynomial extensions if
  interpretability matters.
- **Small n / few events** — overfitting risk high; use
  regularisation, dropout, or shallow models.
- **Non-PH structure** — pure DeepSurv still assumes proportional
  hazards; use Coxtime or DeepHit.

## Assumptions & caveats

- **Proportional hazards** — DeepSurv assumes it in the
  time-invariant form; Coxtime relaxes.
- **Baseline hazard** — the NN estimates the exp(g) factor;
  recover `h_0(t)` via Breslow after fitting.
- **Non-informative censoring** — same as classical survival.
- **Reproducibility** — set seeds; results sensitive to init and
  dropout schedule.
- **Evaluation** — C-index, integrated Brier score, time-dependent
  ROC.

## Related in this repo

- `cox-ph`, `cox-time-varying`, `random-survival-forest`,
  `accelerated-failure-time` — survival family.
- `deep-mlp-backprop`, `neural-ode`, `attention-mechanism` — NN
  building blocks.
- `harrell-c-index`, `discrimination-calibration` — evaluation
  metrics.

## Run

```
python techniques/deep-survival-network/python/deep_survival_network.py
Rscript techniques/deep-survival-network/r/deep_survival_network.R
```

**Refs:** Katzman, J.L. et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." *BMC Medical Research Methodology*, 18: 24, 2018; Lee, C. et al. "DeepHit: a deep learning approach to survival analysis with competing risks." *AAAI*, 2018.

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
