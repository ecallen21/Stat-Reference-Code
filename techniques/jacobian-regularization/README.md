# Jacobian Regularisation (Reference Ch 30 Robustness)

Add a **Frobenius-norm penalty on the input-output Jacobian** so the
network is locally smooth. Hoffman, Roberts & Yaida (2019) — and
independently Jakubovitz & Giryes (2018) — showed the penalty gives
adversarial robustness on par with dedicated defences at a fraction of
the cost.

## Loss

```
L_total(x, y) = L_data(f(x), y)  +  (λ / 2) · ‖ J_f(x) ‖_F²
```

`J_f(x) = ∂f / ∂x` is the `(d_out × d_in)` Jacobian; the Frobenius norm
is the sum of squared entries.

## Cheap Hutchinson estimator (Hoffman 2019)

```
‖ J ‖_F²  =  𝔼_{v ~ N(0, I_{d_out})}  ‖ Jᵀ v ‖²
```

One backward pass per Monte-Carlo draw; unbiased and low variance for
`n_samples = 1–8`.

## Why it helps

- **Local Lipschitz** — small `‖J‖` ⇒ output bounded by input change.
- **Adversarial robustness** — `L_2` attacks have bounded effect.
- **Calibration** — flatter models tend to be less over-confident.
- **Feature disentanglement** — penalises directions of high sensitivity.

## Relation to other regularisers

| Regulariser | What it penalises            | Same-effect knob |
|-------------|------------------------------|------------------|
| L2 weight decay | `Σ ‖W_l‖_F²`               | strong-effect proxy |
| Spectral norm   | `σ_max(W_l)` per layer     | hard Lipschitz cap |
| Jacobian penalty | `‖J_f‖_F² = Σ σ_i(J)²`   | direct smoothness  |
| Contractive AE  | `‖J_encoder‖_F²`           | for representations |

## Files

- `python/jacobian_regularization.py` —
  **Part 1**: analytic `‖J‖_F²` for a 2-layer ReLU MLP compared with
  the Hutchinson estimator (100 samples). `analytic = 4.51`,
  `Hutchinson = 4.18` — unbiased.
  **Part 2**: L2 weight-decay sweep (a strong-effect proxy for the
  penalty) — `wd` from 0 to 0.50; `‖J‖_F²` drops 1.66 → 0.15 and
  empirical Lipschitz 1.41 → 0.39, at a growing train-MSE cost.
- `r/jacobian_regularization.R` — `reticulate` + PyTorch / JAX; native
  R via `numDeriv` for the Jacobian on small nets.

## Assumptions & caveats

- **Cost** — even the Hutchinson estimator adds one backward per sample.
  Practical recipes stochastically apply the penalty only on a mini-
  batch subset.
- **BatchNorm** — the Jacobian through BN depends on batch statistics;
  train + eval Jacobians differ.
- **Softmax output** — for classification, penalise the Jacobian of the
  *logits* not the probabilities to avoid degenerate zeros at saturation.
- **Not certified** — the smoothness is empirical; use
  `randomized-smoothing` when a proof is required.
- **λ is problem-dependent** — start at `1e-3`, tune to trade off
  clean accuracy vs robustness.

## Related in this repo

- `spectral-normalization` — hard per-layer Lipschitz cap.
- `gradient-clipping` — smoothness in the optimiser rather than the
  loss.
- `randomized-smoothing`, `pgd-adversarial-training`, `trades-adversarial`
  — dedicated adversarial defences.
- `dropout-batchnorm`, `deep-mlp-backprop` — the training loop that
  hosts the penalty.
- `contractive-autoencoder` (if present) — Jacobian penalty on the
  *encoder* rather than the classifier.

## Run

```
python techniques/jacobian-regularization/python/jacobian_regularization.py
Rscript techniques/jacobian-regularization/r/jacobian_regularization.R
```

**Refs:** Hoffman, J., Roberts, D.A. & Yaida, S. "Robust learning with Jacobian regularisation." arXiv:1908.02729, 2019; Jakubovitz, D. & Giryes, R. "Improving DNN robustness to adversarial attacks using Jacobian regularisation." *ECCV*, 2018; Rifai, S. et al. "Contractive auto-encoders: explicit invariance during feature extraction." *ICML*, 2011.

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
