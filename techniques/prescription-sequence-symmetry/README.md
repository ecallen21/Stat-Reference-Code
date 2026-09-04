# Prescription Sequence Symmetry Analysis (Reference §43.5, §43.13)

Hallas (1996), Lai et al. (2021). **Within-person** ADR signal
detection: for two drugs A and B, count how many patients started
A before B vs B before A among users of both. Under **no causal
effect** of A on the condition treated by B, the sequence is
symmetric.

## Statistic

- **Crude sequence ratio** `SR = n_AB / n_BA`.
- **Adjusted sequence ratio** `ASR = SR / null_ratio`, correcting
  for time trends in prescribing (if A prescribing rises over the
  study period, more A→B sequences appear even under no causal
  effect).
- **Binomial test** on `n_AB / (n_AB + n_BA)` against a null
  proportion derived from the null ratio.

Signal cutoff: `ASR ≥ 1.2 AND p < 0.05`.

## When to use

- **Automatic hypothesis generation** in claims / prescription
  databases.
- **Rapid screening** — much cheaper than a full cohort study.
- **Bias mitigation** — within-person design removes time-fixed
  confounders.

## When NOT to use

- **Reverse causation** — the condition may cause A first, then B.
- **Non-transient exposures** — chronic drugs blur the ordering.
- **Strong secular trends** — must estimate and adjust the null
  ratio properly.

## Files

- `python/prescription_sequence_symmetry.py` — count A→B vs B→A
  among common users + binomial test with null-ratio adjustment.
  Demo (n=400, injected 70/30 asymmetry): **n_AB = 273, n_BA = 127,
  SR = 2.15, p = 2.3 × 10⁻¹³** — strong signal.
- `r/prescription_sequence_symmetry.R` — custom via
  `survival` + `lubridate`, `Epi::Lexis` (R); `pandas` + `scipy.
  stats.binomtest` (Python).

## Assumptions & caveats

- **Common users** — the analysis is restricted to patients who
  ever received both drugs; ignore this restriction and the
  interpretation collapses.
- **Null ratio estimation** — from historical prescribing trends
  or from a control drug pair; mis-estimation biases ASR.
- **Duration bias** — patients who receive A for a long duration
  are more likely to develop B secondary; sensitivity analyses vary
  the time window.
- **Signal ≠ causation** — PSSA is hypothesis-generating; confirm
  with a formal cohort or self-controlled analysis.

## Related in this repo

- `sccs-self-controlled` — companion within-person design with
  formal IRR.
- `disproportionality-signal-detection` — SRS-based cousin.
- `immortal-time-bias` — an entirely different within-person
  concern.

## Run

```
python techniques/prescription-sequence-symmetry/python/prescription_sequence_symmetry.py
Rscript techniques/prescription-sequence-symmetry/r/prescription_sequence_symmetry.R
```

**Refs:** Hallas, J. "Evidence of depression provoked by cardiovascular medication: a prescription sequence symmetry analysis." *Epidemiology*, 1996; Lai, E.C.-C., Pratt, N.L., Hsieh, C.-Y. et al. "Prescription sequence symmetry analysis: assessing and controlling for prescribing trends and duration biases." *Clinical Epidemiology*, 2021.

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
