# Deep MLP with Back-Propagation (Reference §27.1)

Multi-layer perceptron of arbitrary depth. Extends the 1-hidden-layer
model in `neural-network-mlp` with:

- **Depth**: any number of hidden layers.
- **Non-linearity**: ReLU (default), tanh, GELU, Swish, LeakyReLU.
- **Output head**: softmax + cross-entropy (classification), linear + MSE (regression).
- **Initialisation**: **He** for ReLU (`Var(W) = 2 / fan_in`), **Xavier** for tanh (`Var(W) = 1 / fan_in`).
- **Optimiser**: mini-batch SGD (see `adam-optimizer` for Adam / AdamW).

## Back-propagation

Combined softmax + cross-entropy gradient collapses to `δ^L = ŷ − y`.
For hidden layers `l < L`:

```
δ^l = (W^{l+1})ᵀ δ^{l+1} · φ'(z^l)
∂L/∂W^l = a^{l-1} (δ^l)ᵀ
∂L/∂b^l = δ^l
```

Implemented as a single sweep over layers in reverse order.

## When to use

- **Baseline neural network** on tabular data; feature interactions are learned.
- **Warm start / distillation** target for larger models.
- **Regression / classification / autoencoder inner blocks**.
- **Almost never state-of-the-art alone on images / text / audio** — see `convolutional-nn`, `recurrent-nn`, `transformer-encoder`.

## Files

- `python/deep_mlp_backprop.py` — from-scratch deep MLP with He init, ReLU + softmax-CE, and manual back-prop. Demo (3-class 2D spirals, sizes `[2, 32, 32, 3]`): train accuracy 1.000, test 0.993 in 400 epochs; sklearn `MLPClassifier(hidden=(32,32), solver='adam')` matches at 0.993.
- `r/deep_mlp_backprop.R` — `torch::nn_module`, `keras3`, `nnet::nnet`, `RSNNS::mlp`.

## Assumptions & caveats

- **Vanishing gradients** at depth — He init + ReLU mitigate but don't fix past ~20 layers; use residual connections (`transformer-encoder` uses them) or **batch normalisation** (`dropout-batchnorm`).
- **Dead ReLU units** — dead if pre-activation stays negative; use LeakyReLU or GELU if you see many zero-gradient neurons.
- **Softmax + cross-entropy numerics** — always combine into `log-softmax + NLL` in practice; the combined-gradient trick avoids explicit log(softmax) at all.
- **Batch size vs learning rate** — scale linearly under SGD; more nuanced under Adam. Warmup helps.
- **Regularisation** — L2 weight decay + dropout are cheap and effective; see `dropout-batchnorm`.
- **Full-batch vs mini-batch** — mini-batch SGD converges faster in wall-clock and often generalises better.

## Related in this repo

- `neural-network-mlp` — single-hidden-layer baseline.
- `dropout-batchnorm`, `adam-optimizer`, `embedding-layers` — training add-ons.
- `convolutional-nn`, `recurrent-nn`, `lstm-gru`, `attention-mechanism`, `transformer-encoder`, `autoencoder`, `variational-autoencoder`, `gan-training` — architecture families built on the same forward/backward primitives.

## Run

```
python techniques/deep-mlp-backprop/python/deep_mlp_backprop.py
Rscript techniques/deep-mlp-backprop/r/deep_mlp_backprop.R
```

**Refs:** Rumelhart, D.E., Hinton, G.E. & Williams, R.J. "Learning representations by back-propagating errors." *Nature* 323, 533–536, 1986; He, K. et al. "Delving deep into rectifiers: surpassing human-level performance on ImageNet classification." *ICCV*, 2015; Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning*, MIT Press, 2016.

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
