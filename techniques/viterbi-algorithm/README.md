# Viterbi Algorithm (Reference §47.155)

Viterbi (1967). Dynamic-programming MAP decoder for HMMs / Markov
chains:

    δ_t(j) = max_i δ_{t−1}(i) · a_{ij} · b_j(o_t)
    ψ_t(j) = argmax_i δ_{t−1}(i) · a_{ij}

then backtrack from t = T to recover the most likely state
sequence. O(T · K²), same as forward-backward. Log-space
avoids underflow.

## Files

- `python/viterbi_algorithm.py` — from-scratch log-space Viterbi
  on the classic **fair / loaded die HMM** (Durbin 1998 example)
  with 300 rolls:
  - **Decoding accuracy = 0.647** (chance = 0.5).
  - Fraction of 'loaded' true = 0.47 / decoded = 0.29.
  - log P*(x, z*) = −536.6.
  - Vectorised implementation agrees on all 300 steps.
- `r/viterbi_algorithm.R` — `HMM::viterbi` — same HMM.

## When to use

- **HMM decoding** — POS tagging, speech, gene finding, chromatin
  states, gesture recognition.
- **CRF decoding** — Viterbi is the standard test-time predictor.
- **Convolutional-code / trellis** decoding in communications
  (its original 1967 use).

## When NOT to use

- **Continuous state spaces** — use Kalman / particle smoothers.
- **Long sequences with large K** — O(T · K²) can be prohibitive;
  beam-Viterbi / A* variants exist.
- **When posterior marginals are needed** — Viterbi gives the
  single best path; use forward-backward instead.

## Assumptions & caveats

- **Log-space** is essential — probabilities underflow quickly for
  long sequences.
- **Ties** — argmax may split; deterministic tie-breaking matters
  for reproducibility.
- **First-order Markov** — higher-order → increase K.
- **Best-path vs marginal-max** are different criteria; Viterbi
  gives the joint MAP.

## Related in this repo

- `hmm`, `baum-welch-hmm` — training-side companions.
- `crf-conditional-random-field` — discriminative sibling.
- `beam-search-decoding` — approximate best-first sequence
  search when exact Viterbi is infeasible.
- `state-space-kalman`, `rts-kalman-smoother` — continuous-state
  analogues.

## Run

```
python techniques/viterbi-algorithm/python/viterbi_algorithm.py
Rscript techniques/viterbi-algorithm/r/viterbi_algorithm.R
```

**Refs:** Viterbi, A. J. "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm." *IEEE Trans. Information Theory* 13(2), 1967; Forney, G. D. "The Viterbi algorithm." *Proc IEEE* 61(3), 1973; Durbin, R., Eddy, S., Krogh, A. & Mitchison, G. *Biological Sequence Analysis*, Cambridge, 1998.

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
