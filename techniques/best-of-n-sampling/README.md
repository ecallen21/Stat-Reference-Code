# Best-of-N Sampling (Reference §47.196)

Cobbe et al. (2021); Nakano et al. (2021, WebGPT). Simplest
test-time scaling for LMs:

    1. Draw N i.i.d. samples from p(y | x) with temperature > 0.
    2. Score each with a REWARD MODEL / verifier.
    3. Return the argmax-scored sample.

Provably improves quality: E[max_i R(y_i)] increases in N with
asymptotic growth **√(2 ln N)** for Gaussian rewards. Cheap to
implement; complements RLHF (which changes the sampling
distribution) — best-of-N is a wrapper that keeps the base model
frozen.

## Files

- `python/best_of_n_sampling.py` — two experiments:
  1. MC E[max of N N(0, 1)] vs the theoretical √(2 ln N):
     - N = 4 → **1.03** (theory 1.67); N = 32 → **2.07** (2.63);
       N = 128 → **2.60** (3.11).
  2. Toy 4-digit-number task with reward = −|y − 4242| −100·[y odd]:
     - N = 1: −2518   N = 16: −301   **N = 256: −34** — ~74×
       reduction in expected error.
- `r/best_of_n_sampling.R` — pure-R MC + theoretical scaling.

## When to use

- **Test-time LLM quality boost** without changing the model —
  the simplest inference improvement.
- **When a good reward model / verifier is available** — the RM
  is often cheaper than a full re-generation.
- **RLHF baseline** — best-of-N with an RM lower-bounds the
  quality achievable by PPO / DPO on the same RM.

## When NOT to use

- **When compute per sample is very high** (long generations,
  large models) — N draws multiply cost.
- **When the RM is misspecified** — best-of-N amplifies RM bias
  (reward hacking).
- **Diverse outputs desired** — best-of-N collapses onto the
  single highest-scoring mode.

## Assumptions & caveats

- **N ~ 4-16 typical** for LLMs; beyond that, gains diminish
  and search methods (ToT, MCTS) become preferable.
- **Reward model calibration** matters — a noisy RM makes
  best-of-N pick spurious winners.
- **Temperature > 0** essential; T = 0 makes all draws identical.
- **Combined with process rewards** (PRM) improves reasoning
  tasks specifically.

## Related in this repo

- `rlhf-preferences`, `dpo-direct-preference-optimization` —
  reward-model training cousins.
- `process-reward-model-prm` — dense-reward variant for reasoning.
- `tree-of-thoughts-reasoning`, `beam-search-decoding` —
  structured-search cousins.
- `constitutional-ai` — the RM can be CAI-trained.

## Run

```
python techniques/best-of-n-sampling/python/best_of_n_sampling.py
Rscript techniques/best-of-n-sampling/r/best_of_n_sampling.R
```

**Refs:** Cobbe, K. et al. "Training verifiers to solve math word problems." *arXiv:2110.14168*, 2021; Nakano, R. et al. "WebGPT: Browser-assisted question-answering with human feedback." *arXiv:2112.09332*, 2021; Snell, C., Lee, J., Xu, K. & Kumar, A. "Scaling LLM test-time compute optimally." *arXiv:2408.03314*, 2024.

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
