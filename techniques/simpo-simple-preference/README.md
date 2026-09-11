# SimPO — Simple Preference Optimization (Reference §47.199)

Meng, Xia & Chen (2024). Like ORPO, SimPO removes the reference
model. The reward is **length-normalised** average log probability,
and adds a **target margin γ** to the pair loss:

    r(x, y) = (1 / |y|) · Σ_t log π(y_t | x, y_<t)
    L_SimPO = −log σ(β · (r(x, y_w) − r(x, y_l) − γ))

Length normalisation cures DPO's LENGTH BIAS (chosen tends to be
shorter to raise total logp). The margin γ tightens the preference
gap.

## Files

- `python/simpo_simple_preference.py` — DPO vs SimPO loss on 8
  simulated pairs where chosen tends to be shorter:
  - DPO loss is constant (0.693) — nearly-equal logratios cancel.
  - **SimPO loss avg 1.29** — length-normalised reward exposes
    the true preference gap.
- `r/simpo_simple_preference.R` — no R port; recommends
  `trl.CPOTrainer` (SimPO variant), `princeton-nlp/SimPO`.

## When to use

- **When DPO exhibits length bias** — SimPO explicitly corrects
  it.
- **Reference-free preference tuning** — halves memory vs DPO.
- **When a target margin γ helps** (harder-to-distinguish
  preferences).

## When NOT to use

- **When length matters intrinsically** — SimPO's per-token
  normalisation may hide legitimate length signal.
- **When paired data is very scarce** — SimPO needs enough pairs
  to estimate the margin reliably.
- **When DPO already works** — SimPO's advantage is subtle.

## Assumptions & caveats

- **β (reward scale)** ~ 2.0 typical (higher than DPO).
- **γ (target margin)** ~ 0.5-1.5; too large → training
  instability.
- **Length normalisation** may weaken signal on very short
  responses; length-invariant SimPO variants exist.
- **CPO** (Xu 2024) is a related contrastive method with a KL
  regulariser bringing back a form of reference-model.

## Related in this repo

- `dpo-direct-preference-optimization`,
  `orpo-odds-ratio`, `kto-kahneman-tversky` — sibling
  preference-optimisation methods.
- `rlhf-preferences` — RL-based alternative.
- `best-of-n-sampling`, `process-reward-model-prm` — inference-
  time selection cousins.

## Run

```
python techniques/simpo-simple-preference/python/simpo_simple_preference.py
Rscript techniques/simpo-simple-preference/r/simpo_simple_preference.R
```

**Refs:** Meng, Y., Xia, M. & Chen, D. "SimPO: Simple preference optimization with a reference-free reward." *arXiv:2405.14734*, 2024.

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
