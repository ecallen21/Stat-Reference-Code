# Hidden Markov Model (Reference §13.14)

A latent Markov chain `S_t ∈ {1, ..., K}` emits categorical observations `y_t ∈ {1, ..., M}`:

```
Pr(S_1 = k)               = π_k
Pr(S_t = j | S_{t-1} = i) = A[i, j]
Pr(y_t = m | S_t = k)     = B[k, m]
```

The states are **hidden**; only `y_{1:T}` is observed.

## Three canonical problems

1. **Forward-backward** — evaluate `p(y_{1:T})` and smoothed posteriors `γ_t(k) = Pr(S_t = k | y_{1:T})`.
2. **Viterbi** — most-likely single state path `S_{1:T}^\*` via dynamic programming.
3. **Baum-Welch (EM)** — estimate `(π, A, B)` from `y_{1:T}` alone.

## Relation to nearby methods

- Same math as **Markov-switching model** (`regime-switching-markov`), but with categorical rather than Gaussian emissions.
- Contrast with **plain Markov chain**: there the states are directly observed.

## Files

- `python/hmm.py` — from-scratch forward-backward, Viterbi, and Baum-Welch EM in log-space. Demo on a 2-state / 3-symbol chain recovers transition and emission matrices to ~0.03 and gives 88.8% Viterbi state-decoding accuracy.
- `r/hmm.R` — `HMM::baumWelch` (or `depmixS4::depmix` with `multinomial()`).

## When to use

- Speech, POS tagging, biological sequence analysis (gene finding, ion-channel recordings).
- Activity recognition from wearable sensor data.
- Any regime-switching series where the observations are discrete (event codes, categorical states).

## Assumptions & caveats

- **Fixed K** — pick with AIC / BIC or by cross-validation on held-out likelihood.
- **Label switching** — after EM, sort states by a monotone characteristic (dominant emission symbol, expected duration).
- **Local optima** — use multiple random starts; EM converges to a local mode of the likelihood.
- **Missing observations** — replace `log_B[:, y_t]` with `0` at missing timepoints.

## Run

```
python techniques/hmm/python/hmm.py
Rscript techniques/hmm/r/hmm.R
```

**Refs:** Baum, L.E. & Petrie, T. "Statistical inference for probabilistic functions of finite state Markov chains." *Ann. Math. Stat.* 37(6), 1554–1563, 1966; Rabiner, L.R. "A tutorial on hidden Markov models and selected applications in speech recognition." *Proc. IEEE* 77(2), 257–286, 1989; Cappé, O., Moulines, E. & Rydén, T. *Inference in Hidden Markov Models*, Springer, 2005.

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
