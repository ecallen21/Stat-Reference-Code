# Multi-Layer Perceptron (Reference §27.1)

Feedforward neural network:

```
h_l = φ(W_l h_{l−1} + b_l)         l = 1, ..., L
ŷ   = softmax(W_L h_{L−1} + b_L)   (classification)
       or W_L h_{L−1} + b_L         (regression)
```

Trained by mini-batch SGD with **backpropagation**:

```
loss = cross-entropy (classification) or MSE (regression)
∇ computed by chain rule
W ← W − lr · ∇W
```

Common activations: **ReLU** (default), tanh, sigmoid, GELU (transformers). Regularization: L2 weight decay, dropout, batch norm, early stopping.

## Files

- `python/neural_network_mlp.py` — from-scratch 1-hidden-layer MLP with ReLU + softmax + full-batch SGD. Demo on 3-class blobs: 99.6% training accuracy, matches sklearn `MLPClassifier(hidden_layer_sizes=(32,))` exactly.
- `r/neural_network_mlp.R` — `nnet::nnet` (base); `keras3` / `torch` for production deep learning.

## When to use

- **Nonlinear decision boundaries** without hand-engineered features.
- **Universal approximation** — one hidden layer with enough units can approximate any continuous function (Cybenko 1989).
- **When you have enough data** — MLPs need more training data than trees.

## When to prefer

- **Trees / GBM** for tabular data — usually beat MLPs on structured features.
- **CNNs / transformers** for images / text / sequences.
- **Linear models** when interpretable coefficients matter.

## Hyperparameters

- **Architecture** — width and depth. Start with 1–2 hidden layers of `~n_features`.
- **Learning rate** — 10⁻³ to 10⁻¹ for SGD; 10⁻⁴ to 10⁻³ for Adam.
- **Batch size** — 32–256; small batches regularize, large batches use hardware.
- **Weight init** — He (ReLU) or Glorot (tanh); implemented here as `N(0, √(2/fan_in))`.
- **Dropout** — 0.2–0.5 on hidden layers.

## Assumptions & caveats

- **Standardize inputs** — helps SGD converge.
- **Local optima** — SGD finds a local minimum; multiple random restarts sometimes help.
- **Overfitting** — early stopping + dropout + weight decay + data augmentation.
- **Calibration** — softmax outputs are often overconfident; use temperature scaling (see `calibration-scaling`).

## Run

```
python techniques/neural-network-mlp/python/neural_network_mlp.py
Rscript techniques/neural-network-mlp/r/neural_network_mlp.R
```

**Refs:** Rumelhart, D.E., Hinton, G.E. & Williams, R.J. "Learning representations by back-propagating errors." *Nature* 323, 533–536, 1986; Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning*, MIT Press, 2016.

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
