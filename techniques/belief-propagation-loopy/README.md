# Loopy Belief Propagation (Reference §47.121)

Pearl (1988, tree BP); Weiss (1997) / Yedidia-Freeman-Weiss (2005)
loopy variant. Sum-product on a pairwise Markov random field:

    m_{i→j}^{new}(x_j) ∝ Σ_{x_i} φᵢ(x_i) ψ_{ij}(x_i, x_j)
                        ∏_{k∈N(i)\j} m_{k→i}^{old}(x_i)
    b_i(x_i) ∝ φᵢ(x_i) · ∏_{k∈N(i)} m_{k→i}(x_i).

Exact on trees, approximate on graphs with cycles (variational
justification: fixed points of Bethe free energy).

## Files

- `python/belief_propagation_loopy.py` — from-scratch loopy BP
  on a 3×3 Ising grid. Demo (ferromagnetic J=+1, small field on
  centre, T=2.5):
  - BP P(s=+1) grid: 0.523–0.573
  - Exact enumeration: 0.519–0.560
  Loopy BP matches exact to ~2 decimals.
- `r/belief_propagation_loopy.R` — `gRain`, `bnlearn` (R);
  `pgmpy`, `libDAI`, from-scratch (Python).

## When to use

- **Graphical model inference** — Bayesian networks, MRFs.
- **Turbo / LDPC coding** — loopy BP is the decoder.
- **Stereo vision, image segmentation** — pairwise MRFs.
- **Constraint satisfaction / max-product** — Viterbi is
  max-product BP.

## When NOT to use

- **Highly cyclic dense graphs** — junction-tree or variational
  methods more principled.
- **Deep-network probabilistic reasoning** — VI / MCMC / SGD scale
  better.
- **When exact inference is affordable** — variable elimination /
  jointrees are exact.

## Assumptions & caveats

- **Convergence** not guaranteed on cycles; damp updates or use
  min-sum for MAP.
- **Marginal biases** on loopy graphs — Bethe approximation.
- **Message initialisation** matters — uniform is safe.
- **Free energy** minima ↔ fixed points; gauge choice affects
  numerics.

## Related in this repo

- `bayesian-hierarchical-models`, `gaussian-graphical-model`,
  `causal-discovery-pc`, `notears-dag-learning` — probabilistic
  graphical models.
- `state-space-kalman`, `hmm-baum-welch` (via `state-space-models`)
  — forward-backward is BP on a chain.
- `variational-inference`, `laplace-approximation`,
  `expectation-propagation-ep`, `mcmc-metropolis-hastings`,
  `hmc-nuts` — approximate-inference cousins.
- `stochastic-block-model`, `latent-space-network` — graph-model
  neighbours.

## Run

```
python techniques/belief-propagation-loopy/python/belief_propagation_loopy.py
Rscript techniques/belief-propagation-loopy/r/belief_propagation_loopy.R
```

**Refs:** Pearl, J. *Probabilistic Reasoning in Intelligent Systems.* Morgan Kaufmann, 1988; Yedidia, J.S., Freeman, W.T. & Weiss, Y. "Constructing free-energy approximations and generalized belief propagation algorithms." *IEEE Trans Info Theory* 51(7): 2282-2312, 2005.

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
