# Recurrent Neural Network (Reference §27.3)

Sequence-to-sequence / sequence-to-scalar network with a **hidden state** that
persists across time steps. The Elman cell:

```
h_t = tanh(W_x x_t + W_h h_{t-1} + b_h)
y_t = W_y h_t + b_y
```

Trained by **back-propagation through time (BPTT)** — the network is unrolled
into a feed-forward computation over the sequence, gradients flow backward
through every step.

## Vanishing / exploding gradients

Repeated multiplication by `W_h · tanh'(z)` across many time steps either
shrinks gradients (vanishing) or blows them up (exploding). Standard fixes:

- **Gradient clipping** — clip `‖g‖` at a threshold (implemented in the demo).
- **Truncated BPTT** — unroll only k steps per update.
- **Gated cells** — LSTM / GRU carry a mostly-additive path that preserves gradients (see `lstm-gru`).
- **Attention** — bypasses the sequential bottleneck entirely (see `attention-mechanism`, `transformer-encoder`).

## When to use

- **Short-sequence, few-parameter tasks** — 1-D signal filtering, character-level tagging on windows.
- **Baseline** for sequence classification before reaching for LSTM / transformer.
- **Cheap serving** — single-cell RNN is faster than a transformer for streaming inference.
- **Pedagogy** — the cleanest illustration of temporal weight sharing + BPTT.

## Files

- `python/recurrent_nn.py` — from-scratch Elman RNN + BPTT with gradient clipping. Demo (`last-token = a` sequence classification, sequences of length 4–8): 100% train and test accuracy. A "first-token" variant would exhibit the vanishing-gradient failure that motivates gated cells; see the `lstm-gru` module for the fix.
- `r/recurrent_nn.R` — `torch::nn_rnn`, `keras3::layer_simple_rnn`.

## Assumptions & caveats

- **Sequential** — no parallelisation across time; slower to train than transformers on GPUs for long sequences.
- **Truncated BPTT introduces bias** — long-range dependencies past the truncation window get zero gradient signal.
- **Bidirectional variants** — use both `h_t → h_{t+1}` (forward) and `h_t ← h_{t+1}` (backward) hidden states; only for offline tasks (you need the whole sequence).
- **Deep RNNs** — stack cells vertically (`num_layers > 1`); can also add residual connections.
- **Cell state initialisation** — zero at t = 0 by convention; learnable initial state is a small improvement.
- **Fixed-length input** — pad + mask (PyTorch `pack_padded_sequence`, Keras masking) to avoid learning from padding tokens.

## Related in this repo

- `lstm-gru` — gated cells fix the vanishing-gradient limitation.
- `attention-mechanism`, `transformer-encoder` — modern replacements for many sequence tasks.
- `embedding-layers` — the input representation for token sequences.
- `hmm` — the probabilistic sequence-modelling counterpart.

## Run

```
python techniques/recurrent-nn/python/recurrent_nn.py
Rscript techniques/recurrent-nn/r/recurrent_nn.R
```

**Refs:** Elman, J.L. "Finding structure in time." *Cognitive Science* 14(2), 179–211, 1990; Werbos, P.J. "Backpropagation through time: what it does and how to do it." *Proc. IEEE* 78(10), 1550–1560, 1990; Pascanu, R., Mikolov, T. & Bengio, Y. "On the difficulty of training recurrent neural networks." *ICML*, 2013.

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
