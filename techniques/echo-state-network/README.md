# Echo State Network / Reservoir Computing (Reference §47.310)

Jaeger (2001); Lukoševičius & Jaeger (2009). A large, FIXED,
sparsely-connected recurrent RESERVOIR maps inputs to
high-dimensional states; only the LINEAR READOUT is trained:

```
h_t = tanh(W_in x_t + W_res h_{t−1})    (fixed random weights)
ŷ_t = W_out h_t                          (learn via ridge)
```

Fast to train (no BPTT), competitive with LSTM on many
time-series tasks provided reservoir SPECTRAL RADIUS < 1
("echo state property").

## Files

- `python/echo_state_network.py` — 200-neuron reservoir, ρ=0.9,
  sparsity 0.05. Predict 5-step-ahead on a sinusoidal signal
  (T=3000, 2000 train / 1000 test). ESN test RMSE ≈ 0.03
  vs lag-1-persistence baseline ~1.4. Perturbation decay
  confirms echo-state property (‖h_A − h_B‖ → 0 within 20
  steps).
- `r/echo_state_network.R` — `reservoir` (R), reticulate +
  `reservoirpy` (R); `reservoirpy`, `easyesn`, `pyESN`,
  from-scratch (Python).

## When to use

- **Non-stationary / non-linear time series** where LSTM is
  expensive to train.
- **Online / streaming** — recursive least squares update
  on the readout.
- **Physical reservoirs** — optical, spintronic; ESN theory
  applies.

## When NOT to use

- **Very long-term dependencies** — LSTM / Transformer better
  represent state.
- **Task requires learned representations** — reservoir is
  fixed; can be a bottleneck.
- **Tiny data** — ridge on high-dim reservoir features can
  overfit.

## Assumptions & caveats

- **Spectral radius** — < 1 guarantees echo state property in
  practice; check with `max(|eig(W_res)|)`.
- **Sparsity** — 1-20% typical; sparser is faster.
- **Leaky integration** — h_t = (1−α) h_{t−1} + α tanh(…)
  smooths dynamics.
- **Warmup** — discard first ~ 100 samples when computing
  the readout to let the reservoir settle.

## Related in this repo

- `lstm-gru` — trainable RNN analogues.
- `state-space-kalman`, `state-space-models` — linear
  cousins.
- `mamba-state-space-transformer` — modern SSM alternative.
- `random-fourier-features` — related random-feature idea.

## Run

```
python techniques/echo-state-network/python/echo_state_network.py
Rscript techniques/echo-state-network/r/echo_state_network.R
```

**Refs:** Jaeger, H. "The echo state approach to analysing and training recurrent neural networks." *GMD Report 148*, 2001; Lukoševičius, M. and Jaeger, H. "Reservoir computing approaches to recurrent neural network training." *Computer Science Review*, 3(3): 127-149, 2009.

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
