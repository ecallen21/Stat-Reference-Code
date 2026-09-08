# Contrastive Predictive Coding (Reference §47.99)

van den Oord, Li & Vinyals (2018). Learn representations by
predicting FUTURE samples in the latent space:

    z_t = g_enc(x_t)                    (encoder)
    c_t = g_ar(z_{≤ t})                 (autoregressive summary)
    f_k(x_{t+k}, c_t) = exp(z_{t+k}ᵀ W_k c_t)   (score)

InfoNCE loss:

    L = −E[log  f_k(x_{t+k}, c_t) / Σ_{x* ∈ N ∪ {x_{t+k}}} f_k(x*, c_t) ]

Maximises a variational LOWER BOUND on mutual information I(z_{t+k}, c_t).
Foundation of wav2vec 2.0, MoCo, SimCLR (image variant).

## Files

- `python/contrastive_predictive_coding.py` — linear encoder +
  linear transition, InfoNCE loss with N_neg=8 negatives, from-
  scratch numerical training. Demo (12 sinusoidal sequences with
  distinct frequencies, T=20):
  - Predictive retrieval accuracy (positive > 1 negative)
    ≈ 0.65 after 40 epochs (chance = 0.50).
- `r/contrastive_predictive_coding.R` — no first-class R port;
  `cpc-audio`, `torchaudio`, SimCLR-time in Python.

## When to use

- **Self-supervised representation learning** on sequential data
  — audio (wav2vec), text, sensor streams, video.
- **Pre-training encoders** before a downstream linear probe.
- **When labelled data is scarce** — leverage vast unlabelled
  sequences.

## When NOT to use

- **Small unlabelled corpora** — needs many negatives per anchor.
- **Non-sequential data** — use SimCLR / BYOL image variants.
- **When strong labels are already available** — supervised
  usually beats SSL at high label budgets.

## Assumptions & caveats

- **Number of negatives** — larger N → tighter MI lower bound.
- **Bandwidth of the encoder** — too small loses info; too large
  makes InfoNCE trivial via shortcut features.
- **Predictor horizon k** — longer k → harder task → richer
  features but noisier gradients.
- **Bilinear vs MLP score** — bilinear is provably a valid MI
  bound; MLP scores are heuristic.

## Related in this repo

- `contrastive-learning`, `byol-simsiam`, `jepa-self-supervised`
  — SSL cousins.
- `word-embeddings`, `sentence-similarity`,
  `masked-language-modeling`, `transformer-encoder` — pre-training
  neighbours.
- `mutual-information`, `conditional-mutual-info`,
  `f-divergences`, `kl-divergence` — information-theoretic
  underpinnings.
- `mmd-two-sample-test`, `hsic-independence` — kernel-based
  dependence measures.

## Run

```
python techniques/contrastive-predictive-coding/python/contrastive_predictive_coding.py
Rscript techniques/contrastive-predictive-coding/r/contrastive_predictive_coding.R
```

**Refs:** van den Oord, A., Li, Y. & Vinyals, O. "Representation learning with contrastive predictive coding." *arXiv:1807.03748*, 2018; Baevski, A., Zhou, Y., Mohamed, A. & Auli, M. "wav2vec 2.0: A framework for self-supervised learning of speech representations." *NeurIPS*, 2020.

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
