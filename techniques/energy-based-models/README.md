# Energy-Based Models — EBM (Reference §27.x extra)

Model an unnormalised density via a learned **energy function**:

```
p_θ(x) = exp(−E_θ(x)) / Z(θ),   Z = ∫ exp(−E_θ(x)) dx (intractable)
```

Low energy ↔ high density. `E_θ` can be **any** differentiable neural
network — that's what makes EBMs so flexible.

## Training via contrastive divergence

Maximum-likelihood gradient (Hinton 2002):

```
∂/∂θ log p(x) = − ∂/∂θ E(x)        (positive phase: data)
              + 𝔼_{x' ~ p_θ} [∂/∂θ E(x')]     (negative phase: model samples)
```

Negative phase needs MCMC (Langevin, HMC, Gibbs) from `p_θ`. **Contrastive
divergence CD-k** (Hinton 2002) runs `k` MCMC steps from each data point —
a cheap approximation that works well for RBMs and modern deep EBMs.

## Langevin dynamics

```
x ← x − ε · ∇_x E_θ(x) + √(2ε) · η,   η ~ N(0, I)
```

Long chains converge to `p_θ`. Used for both training (negative phase) and
inference (sampling from a trained EBM).

## Family

- **Restricted Boltzmann Machines** (Hinton) — the historical starting point; CD training.
- **JEM** (Grathwohl 2020) — joint energy-based classifier `p(x, y)` from a discriminative net.
- **Score matching** (Hyvärinen 2005) — trains the gradient `∇ log p(x)` directly; avoids `Z` entirely.
- **Denoising score matching** (Vincent 2011) → **diffusion models** (see `diffusion-model`).
- **NCE** (Gutmann-Hyvärinen 2010) — treat density estimation as classification against a noise distribution.

## When to use

- **Density estimation** where GANs / VAEs' inductive biases don't fit.
- **Anomaly detection** — low-density regions have high energy.
- **Structured prediction** — CRFs are conditional EBMs.
- **Generative modelling** — often superseded by diffusion / flows, but EBM lineage underlies them.
- **NOT** when you need cheap sampling — MCMC per sample is expensive.

## Files

- `python/energy_based_models.py` — from-scratch Gaussian energy `E(x) = ‖x − μ‖² / 2σ²` trained via CD-15 Langevin. Demo (500 samples from `N([2, −1], 0.5 I)`, learn μ with σ²=1 fixed):
  - Learned `μ = [1.94, −1.02]` matches data mean `[1.94, −1.01]` exactly.
  - Long Langevin chain samples correctly hit stationary `N(μ, σ²=1)` — sample mean `[1.97, −1.03]`, sd ≈ 1.0.
- `r/energy_based_models.R` — no strong native support; `reticulate` + Python (roll-your-own in torch / JAX); score-based / diffusion references.

## Assumptions & caveats

- **Partition function `Z`** is never computed — likelihood values are un-normalised. Use bridge sampling or annealed IS for evaluation.
- **CD is biased** — the k-step MCMC doesn't fully mix; CD-1 is fine for RBMs but fails for very sharp densities.
- **Persistent CD** (Tieleman 2008) — keep the MCMC chain across training steps; better negative samples.
- **Langevin step-size / burn-in** matter — too big diverges; too small under-samples the tails.
- **Deep EBMs** need extra tricks (spectral norm, gradient clipping, regularised energy) to keep training stable.
- **Score matching** avoids CD entirely and scales better; diffusion models are its state-of-the-art descendant.

## Related in this repo

- `diffusion-model` — score-based generative model in the EBM lineage.
- `variational-autoencoder`, `gan-training`, `normalizing-flows` — sibling generative families.
- `mcmc-metropolis-hastings`, `gibbs-sampler`, `hamiltonian-mc` — the MCMC neighbours EBMs rely on.

## Run

```
python techniques/energy-based-models/python/energy_based_models.py
Rscript techniques/energy-based-models/r/energy_based_models.R
```

**Refs:** Hinton, G.E. "Training products of experts by minimizing contrastive divergence." *Neural Comput.* 14(8), 1771–1800, 2002; Hyvärinen, A. "Estimation of non-normalized statistical models by score matching." *JMLR* 6, 695–709, 2005; Grathwohl, W. et al. "Your classifier is secretly an energy-based model and you should treat it like one (JEM)." *ICLR*, 2020; Song, Y. & Ermon, S. "Generative modeling by estimating gradients of the data distribution." *NeurIPS*, 2019.

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
