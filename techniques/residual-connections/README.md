# Residual Connections (Reference §27.x extra)

He et al. (2015). The single most important architectural change of the
past decade for training deep networks:

```
y = x + F(x)
```

The identity shortcut lets the **gradient** flow directly through the block:

```
∂L/∂x = ∂L/∂y · (I + ∂F/∂x)
```

If `F` is close to zero at init (proper scaling / LayerScale / zero-init), the
identity path dominates and the gradient survives arbitrary depth.

## Where they appear

- **ResNet-50 / 101 / 152** — CNN classifier backbones.
- **Transformer** — residuals around both self-attention and FFN sublayers.
- **U-Net** — skip connections from encoder to decoder.
- **DenseNet** — concatenative rather than additive skips.
- **Diffusion U-Net** — residuals + attention interleaved.

## Variants

- **Highway networks** (Srivastava 2015) — gated residual `y = t · F(x) + (1 − t) · x`.
- **DenseNet** — concatenate all preceding features.
- **ReZero** (Bachlechner 2020) — `y = x + α · F(x)` with learnable `α` init at 0; guarantees identity at init.
- **LayerScale** (Touvron 2021) — same idea but per-channel scaling in ViT / DINO.
- **DeepNorm** (Wang 2022) — scaling factors that support 1000-layer transformers.

## When to use

- **Any network with > ~10 layers** — residuals are near-mandatory now.
- **CNN classifiers** — ResNet family; the standard starting point.
- **Transformers** — required by every transformer.
- **Diffusion U-Net** — residual + attention interleaved is the modern denoiser architecture.
- **Warm-up long training** — LayerScale / ReZero init help long runs stay stable.

## Files

- `python/residual_connections.py` — from-scratch numpy demo. Forward + backward through 40 stacked blocks (`d=32`, weight scale 0.10) with and without a skip. Gradient-norm propagation from the output back to the input:
  - **Plain stack**: gradient decays to 1.7e-17 at layer 0 (vanishing gradients).
  - **Residual stack**: gradient stays finite (5.3e2 at layer 0), because `∂L/∂x = ∂L/∂y · (I + ...)` gets a free `∂L/∂y` term per layer that never vanishes.
- `r/residual_connections.R` — `torch::nn_module` with `y = x + self$block(x)`, `keras3::layer_add`; standard-arch references (ResNet, Transformer, DenseNet, Highway, ReZero, LayerScale, DeepNorm).

## Assumptions & caveats

- **F must be near-zero at init for the identity path to dominate** — He init on the last layer of F, or an explicit small learnable scale (ReZero / LayerScale) helps.
- **Shape mismatch** at downsampling boundaries — use a 1×1 conv / linear projection on the identity path when dimensions change (ResNet's projection shortcut).
- **Not a substitute for norm** — Residual + LayerNorm (or BatchNorm) are the standard pair; each helps a different failure mode.
- **Pre-norm vs post-norm** in transformers — pre-norm is more stable for depth; both work.
- **Optimiser sensitivity** — Adam / AdamW is the default with residuals; plain SGD needs warmup.

## Related in this repo

- `deep-mlp-backprop`, `convolutional-nn`, `transformer-encoder` — architectures where residuals live.
- `dropout-batchnorm` — the norm counterpart.
- `adam-optimizer` — the training-loop counterpart.
- `lr-schedules` — the LR-schedule counterpart.

## Run

```
python techniques/residual-connections/python/residual_connections.py
Rscript techniques/residual-connections/r/residual_connections.R
```

**Refs:** He, K. et al. "Deep residual learning for image recognition." *CVPR*, 2016; Srivastava, R.K., Greff, K. & Schmidhuber, J. "Training very deep networks." *NeurIPS*, 2015; Huang, G. et al. "Densely connected convolutional networks." *CVPR*, 2017; Bachlechner, T. et al. "ReZero is all you need: fast convergence at large depth." *UAI*, 2021.

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
