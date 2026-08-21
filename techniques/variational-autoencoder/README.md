# Variational Autoencoder — VAE (Reference §27.8)

Probabilistic generative model with a latent variable and an amortised
approximate posterior. Kingma & Welling (2013), Rezende et al. (2014).

## Model

- **Prior**: `z ~ N(0, I)`, `z ∈ ℝ^k`.
- **Likelihood** (decoder): `p_θ(x | z) = N(µ_θ(z), σ_x² I)` for continuous `x`, or Bernoulli for binary.

## Approximate posterior (encoder)

Amortised diagonal Gaussian:

```
q_φ(z | x) = N( µ_φ(x), diag(σ²_φ(x)) )
```

The encoder emits both `µ_φ(x)` and `log σ²_φ(x)` per data point.

## ELBO objective

Maximise the evidence lower bound:

```
ELBO = E_q[log p_θ(x | z)] − KL(q_φ(z | x) || p(z))
```

- **Reconstruction term** — expected log-likelihood of `x` under samples from `q`.
- **KL regulariser** — closed-form for two Gaussians: `½ Σ_k (µ² + σ² − log σ² − 1)`.

## Reparameterisation trick

Pull the stochasticity out of the encoder:

```
z = µ_φ(x) + σ_φ(x) · ε,   ε ~ N(0, I)
```

Gradients now flow through `µ, σ` (differentiable) while `ε` is external
noise. This is the key innovation that made VAE trainable end-to-end by SGD.

## When to use

- **Generative modelling with a learned latent** — sample from `p(z)` and decode.
- **Manifold learning** — a probabilistic non-linear PCA.
- **Semi-supervised learning** — regularise the encoder using unlabelled data.
- **Anomaly detection via reconstruction log-likelihood**.
- **Modern alternatives** — GANs (implicit likelihood, sharper samples), normalising flows (exact likelihood), diffusion (best sample quality but expensive).

## Files

- `python/variational_autoencoder.py` — from-scratch numpy VAE with one-hidden-layer encoder and decoder, diagonal Gaussian q, reparameterisation trick, closed-form KL. Manual back-prop through the full computation graph (no autograd). Demo (N=500 mixture of two Gaussians embedded in ℝ⁵, k=2 latent, 1500 epochs): −ELBO 4.48; encoded latent centred near 0 with sd ≈ 1 (matches the prior); decoder samples reproduce data variance profile (2.34 / 1.32 / 1.34 vs data 3.05 / 1.72 / 1.92).
- `r/variational_autoencoder.R` — `torch::nn_module` with manual reparameterisation, `keras3` Sequential + Sampling layer + `add_loss`, `pyro` / `numpyro` / `tensorflow-probability` for full probabilistic-programming versions.

## Assumptions & caveats

- **Posterior collapse** — decoder can learn to ignore `z` (KL → 0). Mitigate with KL warmup, β-VAE (`β < 1`), free-bits, InfoVAE.
- **Blurry samples** — Gaussian likelihood + ELBO training favours mean-covered reconstructions; GANs / diffusion give sharper outputs.
- **Diagonal q(z | x)** is restrictive — normalising-flow posteriors or full-covariance heads capture correlated latents.
- **σ_x (observation noise) is a hyperparameter** — too small overweights reconstruction and collapses the KL; too large loses signal.
- **Local optima** — random init + SGD gets stuck; multiple restarts, KL annealing, and Adam (see `adam-optimizer`) all help.
- **Evaluation is hard** — ELBO is a lower bound; IWAE gives a tighter estimator (Burda et al. 2015).

## Related in this repo

- `autoencoder` — deterministic non-probabilistic cousin.
- `gan-training`, `dirichlet-process-mixture` — alternative generative approaches.
- `variational-inference` — the general framework the VAE belongs to.
- `laplace-approximation` — different approximate-posterior family.

## Run

```
python techniques/variational-autoencoder/python/variational_autoencoder.py
Rscript techniques/variational-autoencoder/r/variational_autoencoder.R
```

**Refs:** Kingma, D.P. & Welling, M. "Auto-encoding variational Bayes." *ICLR*, 2014; Rezende, D.J., Mohamed, S. & Wierstra, D. "Stochastic backpropagation and approximate inference in deep generative models." *ICML*, 2014; Doersch, C. "Tutorial on variational autoencoders." *arXiv:1606.05908*, 2016.

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
