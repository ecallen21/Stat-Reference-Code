# LSTM and GRU (Reference §27.4)

Gated recurrent cells that fix the plain RNN's **vanishing-gradient** problem
via a mostly-additive state path.

## LSTM (Hochreiter & Schmidhuber 1997)

```
i_t = σ(W_i [x_t; h_{t-1}] + b_i)          input gate
f_t = σ(W_f [x_t; h_{t-1}] + b_f)          forget gate
o_t = σ(W_o [x_t; h_{t-1}] + b_o)          output gate
g_t = tanh(W_g [x_t; h_{t-1}] + b_g)       candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t             cell state (additive)
h_t = o_t ⊙ tanh(c_t)                       hidden state
```

`c_t` is the crucial additive path — gradients flow along it almost unchanged.

## GRU (Cho et al. 2014)

```
z_t = σ(W_z [x_t; h_{t-1}])                 update gate
r_t = σ(W_r [x_t; h_{t-1}])                 reset gate
h̃_t = tanh(W_h [x_t; r_t ⊙ h_{t-1}])
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t       convex mix
```

Fewer parameters than LSTM, no separate cell state. Comparable in practice.

## When to use

- **Medium-length sequences** — LSTMs still dominate speech recognition, financial time series with local structure, and low-latency streaming.
- **When transformers are overkill** — small datasets, limited compute, streaming.
- **Sequence generation** — character-level LSTM was the classical language-model baseline.
- **Modern default for long sequences** — transformer (see `transformer-encoder`); LSTMs remain competitive for causal / streaming tasks where full-sequence attention is expensive.

## Files

- `python/lstm_gru.py` — from-scratch LSTM and GRU forward passes. Demo: propagate an initial input spike through T=30 steps of a plain RNN, an LSTM, and a GRU with random weights; compare hidden-state norm across time. **RNN loses 10^17 of the signal (1.58e-17 ratio); LSTM preserves 7.16e-04** — the additive cell path buys three orders of magnitude at 30 steps. GRU sits between (initialisation-dependent).
- `r/lstm_gru.R` — `torch::nn_lstm / nn_gru`, `keras3::layer_lstm / layer_gru`.

## Assumptions & caveats

- **Forget-bias initialisation** — standard trick: set `b_f = 1` at init so `f_t ≈ 1` and the cell state passes through unchanged, giving strong initial gradient flow.
- **LSTM vs GRU** — differences are usually small (< 1% on standard tasks); pick based on library convenience.
- **Peephole connections** — a common LSTM variant lets gates see the cell state directly; small but consistent boost on some tasks.
- **Stacked cells** — depth of 2–4 gives most of the improvement; deeper needs residual connections.
- **Bidirectional** — for offline sequence tasks, run forward and backward passes and concatenate. Not usable for causal / streaming.
- **Attention on top** — decoder LSTMs + attention over encoder outputs (Bahdanau) was the pre-transformer SOTA for seq2seq; transformers now dominate but attention-on-LSTM is a cheap upgrade when transformers are too heavy.

## Related in this repo

- `recurrent-nn` — the plain RNN whose limitation these cells fix.
- `attention-mechanism`, `transformer-encoder` — the modern replacement for many sequence tasks.
- `embedding-layers` — input representation.
- `dropout-batchnorm` — LSTM-friendly dropout variants (variational dropout, zoneout).

## Run

```
python techniques/lstm-gru/python/lstm_gru.py
Rscript techniques/lstm-gru/r/lstm_gru.R
```

**Refs:** Hochreiter, S. & Schmidhuber, J. "Long short-term memory." *Neural Comput.* 9(8), 1735–1780, 1997; Cho, K. et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." *EMNLP*, 2014; Greff, K. et al. "LSTM: A search space odyssey." *IEEE Trans. Neural Netw. Learn. Sys.* 28(10), 2222–2232, 2017.

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
