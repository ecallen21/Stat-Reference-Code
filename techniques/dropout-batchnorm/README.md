# Dropout and BatchNorm (Reference §27.10)

Two standard neural-network regularisers / accelerators.

## Dropout (Srivastava et al. 2014)

Randomly zero out a fraction `p` of activations during training; leave the
network untouched at evaluation.

```
train: y = mask ⊙ x / (1 − p),   mask_i ~ Bernoulli(1 − p)
eval : y = x
```

The `1 / (1 − p)` **inverted-dropout** scaling makes `E[y] = E[x]` so eval-mode needs no extra rescaling. Injects noise → regulariser; also implicitly averages an ensemble of thinned sub-networks.

## Batch normalisation (Ioffe & Szegedy 2015)

Standardise each feature per mini-batch:

```
train: mu_B = mean_B(x),   var_B = var_B(x)
       x̂  = (x − mu_B) / √(var_B + ε)
       y  = γ ⊙ x̂ + β                     γ, β learnable
eval : use exponentially-moving-averaged (mu, var) from training
```

- **Accelerates** training by removing internal covariate shift and enabling higher learning rates.
- **Regularises** implicitly via the noise from mini-batch statistics.
- **Learnable γ, β** let the network recover the identity if it wants.

## Variants and when to use each

| Norm | Averages over | Best for |
|---|---|---|
| **BatchNorm** | across the batch, per feature | CNNs with large batches |
| **LayerNorm** | across features, per sample | Transformers, RNNs, small batches |
| **GroupNorm** | across a group of channels | Small-batch training (segmentation, video) |
| **RMSNorm** | variance-only LayerNorm | Modern LLMs (LLaMA); slightly faster |
| **InstanceNorm** | per-sample per-channel | Style-transfer networks |

## When to use

- **BatchNorm** — CNN classifiers, GAN generators (older architectures).
- **LayerNorm** — every transformer, most RNN cells, when batches are small.
- **Dropout** — MLPs and CNNs; typically **not** in transformers on residual paths (interacts badly with LayerNorm); **variational dropout** for RNNs to apply the same mask across time.
- **Together with weight decay** — orthogonal effect; both cheap.

## Files

- `python/dropout_batchnorm.py` — from-scratch dropout (inverted scaling) + BatchNorm (with running mean / var). Demo:
  - Dropout p=0.5 on unit-Gaussian input: fraction zeroed 0.494 (matches p); Var(y_train) ≈ 1.98 due to `1/(1-p)` scaling; eval-mode passes through unchanged.
  - BatchNorm on N(3, 5²) input after 50 training passes: output mean/sd per feature = 0.0 / 1.0 exactly; running-mean / running-var track input statistics; eval-mode uses running stats.
- `r/dropout_batchnorm.R` — `torch::nn_dropout / nn_batch_norm1d / nn_layer_norm / nn_group_norm`, `keras3::layer_dropout / layer_batch_normalization`.

## Assumptions & caveats

- **BatchNorm and small batches** — statistics become noisy; use GroupNorm or LayerNorm instead.
- **BatchNorm at test time** — bugs from wrong train/eval mode are the single most common neural-net bug. Always set `.eval()` for inference.
- **Dropout after ReLU vs before** — usually after ReLU on the output of the activation.
- **Dropout inside residual blocks** may interact with LayerNorm; safer on the FFN branch than on the attention branch.
- **BatchNorm sees leakage** in some settings — batch statistics correlate with labels in class-imbalanced batches, causing subtle information leakage between examples.
- **Fine-tuning** frozen BN layers with unfrozen upstream layers requires care — freeze the running stats and set `.eval()` mode on BN.

## Related in this repo

- `deep-mlp-backprop`, `convolutional-nn`, `transformer-encoder` — the architectures where these are used.
- `adam-optimizer` — the training-loop counterpart.
- `ridge-lasso-elasticnet` — explicit-penalty alternative.

## Run

```
python techniques/dropout-batchnorm/python/dropout_batchnorm.py
Rscript techniques/dropout-batchnorm/r/dropout_batchnorm.R
```

**Refs:** Srivastava, N. et al. "Dropout: a simple way to prevent neural networks from overfitting." *JMLR* 15, 1929–1958, 2014; Ioffe, S. & Szegedy, C. "Batch normalization: accelerating deep network training by reducing internal covariate shift." *ICML*, 2015; Ba, J.L., Kiros, J.R. & Hinton, G.E. "Layer normalization." *arXiv:1607.06450*, 2016.

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
