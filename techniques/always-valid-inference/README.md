# Always-Valid Inference for A/B Tests (Reference §44.4, §44.12)

Johari-Koomen-Pekelis-Walsh (2017), Howard-Ramdas-McAuliffe-Sekhon
(2021). Continuous monitoring of A/B tests inflates type-I error
under fixed-horizon methods ("peeking bias"). Two peek-safe
alternatives:

- **mSPRT (mixture SPRT)** — Robbins' mixture likelihood-ratio
  test with a prior on the effect size yields an always-valid
  p-value at every time `t`:
  ```
  P(any t: p_t ≤ α under H₀) ≤ α
  ```
- **Confidence sequences (CS)** — time-uniform CI with
  `P(μ ∈ CS_t for all t) ≥ 1 − α`.

## When to use

- **Continuous monitoring dashboards** where stakeholders peek at
  intermediate results.
- **Early stopping** for either winners or losers with valid
  inference guarantees.
- **Guardrail monitoring** — any-time-valid stopping-for-harm.

## When NOT to use

- **Fixed-horizon experiments** with no peeking — sequential
  methods are more conservative than needed.
- **Very small effect sizes** — always-valid methods sacrifice
  power for validity; a well-planned fixed-n test wins.

## Files

- `python/always_valid_inference.py` — mSPRT p-values +
  Howard-Ramdas Gaussian confidence sequence (custom). Demo (2000
  sequential observations, α=0.05, τ²=0.5): under H₀ the mSPRT
  rejection rate stays at 0; under H₁ (effect 0.10) mSPRT rejects
  in **11 %** of peek points while naive fixed-horizon rejects
  in **48 %** — naive would over-fire on many peek moments.
- `r/always_valid_inference.R` — `gsDesign::gsDesign`, `rpact`,
  `ldbounds` (R); `sequential-testing`, `confidence-sequence`
  (Python).

## Assumptions & caveats

- **Mixing prior τ²** trades power at small effects for power at
  large effects; select from historical effect-size distributions.
- **Peeking cost** — always-valid methods require ~ 1.5-2× more
  data than fixed-horizon at matched power.
- **Report the stopping rule and prior** — reviewers need to see
  the guarantee holds.
- **Group-sequential vs anytime-valid** — group-sequential (O'Brien-
  Fleming, Pocock) require pre-specified peek times; mSPRT and CS
  do not.

## Related in this repo

- `ab-test-fundamentals`, `mde-sample-size` — fixed-horizon
  baselines.
- `sequential-analysis` — the SPC-side classical SPRT.
- `guardrail-monitoring` — the operational use case.

## Run

```
python techniques/always-valid-inference/python/always_valid_inference.py
Rscript techniques/always-valid-inference/r/always_valid_inference.R
```

**Refs:** Johari, R., Koomen, P., Pekelis, L., & Walsh, D. "Peeking at A/B tests: why it matters, and what to do about it." *KDD*, 2017; Howard, S.R., Ramdas, A., McAuliffe, J., & Sekhon, J. "Time-uniform, nonparametric, nonasymptotic confidence sequences." *Annals of Statistics*, 2021.

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
