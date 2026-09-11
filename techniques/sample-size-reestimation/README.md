# Sample Size Re-estimation (Reference §47.260)

Wittes & Brittain (1990); Cui, Hung & Wang (1999). At an
interim look, re-estimate the required N based on OBSERVED
variance (blinded, preserves α automatically) or effect size
(unblinded, needs Cui-Hung-Wang re-weighting to keep α):

- **Blinded SSR**: pool arms, estimate `σ̂`, plug back into
  the SS formula.
- **Unblinded SSR (CHW)**: `Z_final = √w_1 Z_1 + √w_2 Z_2`
  with pre-specified weights; preserves α regardless of the
  re-estimated `n_2`.

## Files

- `python/sample_size_reestimation.py` — Both procedures.
  Demo: planned δ=0.5, σ_planned=1.0 gives n=63/arm at 80%
  power; true σ=1.4 needs n=124. Blinded SSR at n₁=30 recovers
  σ̂=1.35 and revised n=114. CHW under H₀ over 20 000
  simulations with random `n_2` in [40, 200] preserves Type-I
  at 0.0496 (nominal 0.05).
- `r/sample_size_reestimation.R` — `rpact`, `gsDesign`,
  `adaptTest`, `MAMS` (R); rpact via reticulate, from-scratch
  (Python).

## When to use

- **Uncertain variance / event rate** — internal-pilot SSR
  after a small first stage.
- **Adaptive phase-III trials** — CHW keeps α while allowing
  N to change based on interim signal.
- **Regulatory-friendly** blinded SSR is preferred where
  possible (no alpha implications).

## When NOT to use

- **Small enrichment factor** — SSR helps most when the
  planning parameter is genuinely uncertain (2× or more).
- **Very early interim** (t < 0.1) — variance estimate is
  too noisy; re-estimation adds volatility rather than
  precision.
- **Where re-estimation opens un-blinding risk** without a
  strict firewall — use blinded procedures.

## Assumptions & caveats

- **Pre-specification** — weights (`w_1`, `w_2`) and rules
  MUST be pre-specified in the SAP, not chosen after seeing
  data.
- **Blinded SSR** works cleanly for continuous outcomes;
  binary requires care about pooled event-rate estimation.
- **Weighted combination** loses efficiency slightly relative
  to fixed N — a well-planned trial without SSR is more
  powerful IF the planning parameter is correct.
- **CHW alternatives** — inverse-normal combination
  (Lehmacher-Wassmer) is closely related.

## Related in this repo

- `group-sequential-design`, `alpha-spending-lan-demets` —
  natural companions.
- `conditional-power-futility` — often the trigger for SSR.
- `mde-sample-size` — pre-trial SS calculation.

## Run

```
python techniques/sample-size-reestimation/python/sample_size_reestimation.py
Rscript techniques/sample-size-reestimation/r/sample_size_reestimation.R
```

**Refs:** Wittes, J. and Brittain, E. "The role of internal pilot studies in increasing the efficiency of clinical trials." *Stat. Med.*, 9(1-2): 65-72, 1990; Cui, L., Hung, H.M.J. and Wang, S.-J. "Modification of sample size in group sequential clinical trials." *Biometrics*, 55(3): 853-857, 1999.

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
