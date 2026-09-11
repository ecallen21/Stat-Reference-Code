# Baum-Welch Algorithm (Reference §47.156)

Baum & Petrie (1966); Baum et al. (1970). EM algorithm for
HMMs. **E-step**: forward-backward gives per-state posteriors
γ_t and transition posteriors ξ_t. **M-step**: re-estimate π,
A, B by weighted counts:

    π_i     ← γ_1(i)
    A_ij    ← Σ_t ξ_t(i, j) / Σ_t γ_t(i)
    B_j(v)  ← Σ_{t: o_t = v} γ_t(j) / Σ_t γ_t(j)

Non-decreasing likelihood; converges to a local optimum. Scaling
factors in the α-recursion avoid underflow.

## Files

- `python/baum_welch_hmm.py` — from-scratch Baum-Welch (with
  α-scaling) on the classic **fair / loaded die HMM**, T = 800.
  Learned params (after permutation alignment):
  - Truth A = [[0.9, 0.1], [0.2, 0.8]]
  - Estim A ≈ [[0.71, 0.29], [0.12, 0.88]]
  - Truth P(6 | loaded) = 0.50, Estim ≈ 0.35 — loaded state clearly
    identified by its higher P(6).
  - Log-lik increases monotonically over 100 iters
    (−1922 → −1408).
- `r/baum_welch_hmm.R` — `HMM::baumWelch` for the same fit.

## When to use

- **Unsupervised HMM training** — no state labels, only
  observations.
- **Speech / gene / activity / gesture segmentation** — classic
  applications.
- **Warm-starting a CRF** with unsupervised HMM parameters.

## When NOT to use

- **Fully labelled state sequences** — supervised MLE (counting) is
  simpler and doesn't get stuck in local optima.
- **Continuous states** — use Kalman-EM / EM for LDS instead.
- **Very long sequences with large K** — O(T · K²) per EM step
  can be costly.

## Assumptions & caveats

- **Local optima** — restart from multiple random inits; take the
  best log-likelihood.
- **State-label permutation invariance** — the K states are
  unlabelled; align to truth (or a reference) after training.
- **Scaling / log-space** essential to avoid underflow.
- **Model selection** (choosing K) via AIC / BIC / cross-
  validation.

## Related in this repo

- `viterbi-algorithm`, `hmm` — decoding + inference companions.
- `state-space-kalman`, `unscented-kalman-filter`,
  `ensemble-kalman-filter`, `rts-kalman-smoother` — continuous-
  state EM cousins (Kalman-EM).
- `mixture-models` — Baum-Welch specialises to GMM-EM when T = 1.

## Run

```
python techniques/baum-welch-hmm/python/baum_welch_hmm.py
Rscript techniques/baum-welch-hmm/r/baum_welch_hmm.R
```

**Refs:** Baum, L. E. & Petrie, T. "Statistical inference for probabilistic functions of finite state Markov chains." *Annals of Math Stat* 37(6), 1966; Baum, L. E., Petrie, T., Soules, G. & Weiss, N. "A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains." *Annals of Math Stat* 41(1), 1970; Rabiner, L. R. "A tutorial on hidden Markov models and selected applications in speech recognition." *Proc IEEE* 77(2), 1989.

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
