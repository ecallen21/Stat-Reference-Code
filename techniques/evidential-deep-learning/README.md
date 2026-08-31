# Evidential Deep Learning (Reference Ch 29 UQ)

Predict the parameters of a **higher-order distribution** over the
target's own distribution. Sensoy, Kaplan & Kandemir (2018) for
classification (Dirichlet head); Amini et al. (2020) for regression
(Normal-Inverse-Gamma head). One forward pass gives point prediction
**and** aleatoric + epistemic uncertainty — no sampling, no ensemble.

## Classification (Dirichlet)

The network outputs non-negative **evidence** `e_k ≥ 0` per class; the
Dirichlet parameters are `α_k = e_k + 1` (prior of 1 = uniform). Read-off
formulas:

```
S = Σ_k α_k                        (Dirichlet strength)
p̂_k     = α_k / S                  (categorical mean = point prediction)
vacuity = K / S                     (aleatoric-plus-epistemic; ↓ with evidence)
```

- `S = K` (no evidence) ⇒ **flat prior** ⇒ vacuity `= 1`.
- `S ≫ K` (lots of evidence) ⇒ **sharp categorical** ⇒ vacuity `→ 0`.

## Regression (Normal-Inverse-Gamma)

Head outputs `(γ, ν, α, β)`; the target is modelled as `y ~ N(μ, σ²)` with
`μ ~ N(γ, σ² / ν)` and `σ² ~ Γ⁻¹(α, β)`. Analytic pointers:

```
mean      = γ
aleatoric = β / (α − 1)                (data noise)
epistemic = β / (ν · (α − 1))          (knowledge about mean)
```

## Loss (Sensoy 2018 eq 5-6, classification)

```
L_i =  Σ_k (y_ik − p̂_ik)²  +  p̂_ik (1 − p̂_ik) / (S_i + 1)
       +  λ · KL( Dir(α̃_i)  ‖  Dir(1) )
```

where `α̃_i` zeroes-out the evidence assigned to the true class — the KL
penalises *wrong* evidence but not correct evidence.

## When to use

- **Single-model, single-pass uncertainty** — no ensemble, no MC dropout.
- **Latency-sensitive deployment** (self-driving, edge devices).
- **Downstream selective prediction** — abstain when `vacuity` is high.

## When NOT to use

- **OOD detection on ReLU features** — a well-known failure mode: distant
  inputs can accumulate large evidence (Meinke & Hein 2020, Ulmer 2021).
  Use **distance-aware features** (SNGP, DDU) or combine with an
  ensemble.
- **Regression with heavy tails** — the NIG marginal is Student-t, but the
  degrees of freedom are shaped by `α`; heavy shifts break the fit.

## Files

- `python/evidential_deep_learning.py` —
  **Part 1**: Dirichlet analytics on hand-set `α` (flat, weak, strong,
  conflict). **Part 2**: from-scratch MLP + softplus evidence head +
  evidential MSE loss on synthetic 3-class blobs. Loss drops 0.48 → 0.23;
  in-distribution vacuity `K/S ≈ 0.29`; 9/12 test-blob classifications
  correct.
- `r/evidential_deep_learning.R` — `reticulate` + `evidential-deep-
  learning-pytorch` / `edl-pytorch` / `evidential_regression`.

## Assumptions & caveats

- **ReLU features extrapolate** — vacuity does not always rise on OOD;
  see caveat above.
- **KL penalty strength `λ`** matters — too small ⇒ overconfident;
  too large ⇒ evidence collapses.
- **Correctness of the Dirichlet analytics** does not depend on the
  network; the demo's Part 1 shows the closed-form vacuity behaviour
  cleanly.
- **The classification MSE loss** in Sensoy 2018 is a variational bound
  on categorical NLL; alternative losses (Malinin 2018 Prior Networks)
  exist.

## Related in this repo

- `mc-dropout`, `deep-ensembles`, `bayesian-neural-network` — Bayesian
  alternatives.
- `ood-detection` — evidential vacuity is one common OOD signal.
- `selective-prediction` — abstain when Dirichlet vacuity crosses a threshold.
- `epistemic-aleatoric` — the decomposition made explicit for the NIG head.

## Run

```
python techniques/evidential-deep-learning/python/evidential_deep_learning.py
Rscript techniques/evidential-deep-learning/r/evidential_deep_learning.R
```

**Refs:** Sensoy, M., Kaplan, L. & Kandemir, M. "Evidential deep learning to quantify classification uncertainty." *NeurIPS*, 2018; Amini, A. et al. "Deep evidential regression." *NeurIPS*, 2020; Meinke, A. & Hein, M. "Towards neural networks that provably know when they don't know." *ICLR*, 2020.

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
