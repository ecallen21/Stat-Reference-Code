# Hopfield Network (Reference §47.90)

Hopfield (1982). Recurrent binary network for content-addressable
memory. Store P patterns `ξ^k ∈ {−1, +1}^N` via Hebbian outer-product:

    W = (1/N) Σ_k ξ^k (ξ^k)ᵀ,   W_ii = 0.

Asynchronous update: `s_i ← sign(Σ_j W_ij s_j)`. Converges to a
local energy minimum. Classical capacity ≈ 0.14 N patterns before
spurious minima dominate (Amit-Gutfreund-Sompolinsky 1985). Modern
continuous-Hopfield (Ramsauer et al 2020) achieves exponential capacity.

## Files

- `python/hopfield_network.py` — from-scratch Hebbian storage +
  async recall. Demo (N=100, capacity limit 0.14·N ≈ 14 patterns):
  - P=5,  30% bits flipped → 5/5 exact recovery
  - P=10, 30% flipped     → 4/10 (approaching capacity)
  - P=15, 30% flipped     → 4/15
  - P=20, 30% flipped     → 1/20 (past capacity, spurious minima).
- `r/hopfield_network.R` — limited R; Python `hflayers` (modern
  continuous Hopfield), from-scratch here.

## When to use

- **Associative / content-addressable memory** — recover full
  pattern from partial cue.
- **Model of biological memory** — motivating example in
  computational neuroscience.
- **Modern attention interpretation** — continuous Hopfield =
  self-attention (Ramsauer 2020).
- **Combinatorial optimisation** — TSP / vertex cover via energy
  minimisation.

## When NOT to use

- **Storing more than ~0.14 N patterns** in classical form —
  spurious minima dominate; use modern Hopfield.
- **Real-valued patterns** — classical form binary; modern extension
  handles continuous.
- **Very high-dim images / audio** — direct storage impractical;
  use as an attention layer instead.

## Assumptions & caveats

- **Bipolar encoding** ({−1, +1}) is standard; binary {0, 1} also
  works with rescaling.
- **Async vs sync updates** — async guaranteed to converge;
  synchronous can oscillate.
- **Spurious minima** at ~P/2 mixtures of stored patterns.
- **Storage capacity** is 0.14 N for Hebbian; up to ~N/(2 log N)
  with pseudo-inverse rule.

## Related in this repo

- `energy-based-models`, `contrastive-learning`, `byol-simsiam`,
  `contrastive-predictive-coding` — energy / SSL cousins.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder` — modern-Hopfield-as-attention interpretation.
- `bayesian-neural-network`, `mc-dropout`, `deep-ensembles` —
  Bayesian NN alternatives.
- `recurrent-nn`, `lstm-gru`, `mamba-state-space-transformer` —
  recurrent-net neighbours.

## Run

```
python techniques/hopfield-network/python/hopfield_network.py
Rscript techniques/hopfield-network/r/hopfield_network.R
```

**Refs:** Hopfield, J.J. "Neural networks and physical systems with emergent collective computational abilities." *PNAS* 79(8): 2554-2558, 1982; Ramsauer, H. et al. "Hopfield networks is all you need." *ICLR*, 2021.

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
