# Spectral Normalisation (Reference Ch 30 Robustness)

Constrain each weight matrix to have **spectral norm** `σ(W) ≤ cap` by
dividing by its top singular value. Miyato et al. (2018) proposed it for
stable GAN training; it also gives a Lipschitz bound on the whole
network — a smoothness prior that helps against adversarial and OOD
inputs.

## Constraint

```
W̄  =  W / σ(W)          (cap = 1)
```

`σ(W) = max_v ‖W v‖ / ‖v‖` is the operator 2-norm.

## Cheap estimate: power iteration (Miyato)

Maintain a single left singular vector `u` per layer; update it once
each SGD step:

```
v ← Wᵀ u / ‖Wᵀ u‖
u ← W v  / ‖W v‖
σ̂ ← uᵀ W v
```

Only one matrix-vector product per side — an **O(1)** cost per training
step. Warmup a few dozen iterations at initialisation for accuracy.

## Effect

- **Lipschitz constant** of an L-layer ReLU MLP is bounded by
  `∏_l σ(W_l)` when every layer is spectral-normalised.
- **Small input perturbations** ⇒ bounded output perturbations ⇒ modest
  adversarial robustness (`L_p` attack gradients are bounded).
- **GAN training stability** — the original use case; replaces WGAN-GP
  as the default modern trick.
- **Distance-aware last layer** (SNGP, Liu 2020) uses spectral
  normalisation to prevent feature collapse for OOD detection.

## When to use

- **GAN discriminator** — the default recipe.
- **Any model where Lipschitz smoothness helps**: robust classification,
  uncertainty quantification, energy-based models.
- **In place of weight normalisation** — SN gives a tighter Lipschitz
  bound.

## When NOT to use

- **Highly expressive tasks** — a strict cap can under-fit; use a larger
  cap or reserve SN for the final layers.
- **Convolutions** — full SN of Conv needs a per-shape SVD; the
  literature uses low-rank / truncated approximations.

## Files

- `python/spectral_normalization.py` —
  1. `power_iteration` verified against `numpy.linalg.svd`
     (`|diff| = 2e-12`).
  2. 3-layer MLP with per-layer SN. **Empirical Lipschitz drops from
     1.05 → 0.08** at `cap = 1.0`; grows to 8.6 at `cap = 5.0`.
- `r/spectral_normalization.R` — `reticulate` + `torch.nn.utils
  .parametrize.register_parametrization`; native R via `base::svd`.

## Assumptions & caveats

- **Empirical Lipschitz can be lower than the SN cap** because ReLU
  activations shrink the effective gradient (see demo).
- **Cap × depth** — for L layers with cap `c`, the network's Lipschitz
  is bounded by `c^L`; use `c = 1` to keep the bound at 1.
- **Convolutions** — decompose `W_conv` as a Toeplitz matrix; libraries
  provide approximate SN.
- **BatchNorm** breaks the Lipschitz guarantee — replace with
  weight normalisation or GroupNorm.
- **Ensures worst-case bound** — average-case behaviour usually retains
  much of the base network's expressiveness.

## Related in this repo

- `jacobian-regularization` — a smoothness prior via input-Jacobian
  penalty rather than a hard spectral cap.
- `gradient-clipping` — a related smoothness prior in the *optimiser*.
- `randomized-smoothing`, `pgd-adversarial-training` — dedicated
  robustness techniques that stack with SN.
- `dropout-batchnorm` — the layer family SN often replaces or augments.
- `bayesian-neural-network`, `last-layer-bayesian` — SN is a common
  pre-processing for distance-aware last-layer methods.

## Run

```
python techniques/spectral-normalization/python/spectral_normalization.py
Rscript techniques/spectral-normalization/r/spectral_normalization.R
```

**Refs:** Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. "Spectral normalisation for generative adversarial networks." *ICLR*, 2018; Liu, J. et al. "Simple and principled uncertainty estimation with deterministic deep learning via distance awareness (SNGP)." *NeurIPS*, 2020.

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
