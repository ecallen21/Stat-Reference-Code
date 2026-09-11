# KTO — Kahneman-Tversky Optimization (Reference §47.197)

Ethayarajh, Xu, Muennighoff, Jurafsky & Kiela (2024). Unlike DPO
which needs PAIRED (chosen, rejected) preferences, KTO uses
**unpaired** labels: {desirable} vs {undesirable} rollouts only.

    L_KTO(x, y, label) = 1 − v(β · (log π(y|x) − log π_ref(y|x)))

with v a Kahneman-Tversky value function (concave for gains,
convex for losses, with **loss aversion**). Requires ~10× less
data than DPO because unpaired labels are much easier to collect
(thumbs-up / thumbs-down vs "which is better?").

## Files

- `python/kto_kahneman_tversky.py` — toy KTO on a 5-arm response
  distribution with 4 liked / 3 disliked examples:
  - Reference policy: [0.15, 0.20, 0.15, 0.25, 0.25]
  - KTO-trained: [0.01, 0.34, 0.06, 0.30, 0.30]
  - **Liked prob mass 70 % → 93 %**; disliked 30 % → 7 %.
- `r/kto_kahneman_tversky.R` — no R port; recommends
  `trl.KTOTrainer`, `ContextualAI/HALOs`.

## When to use

- **Thumbs-up / thumbs-down** production feedback (much cheaper
  than pairwise annotation).
- **Alignment with lopsided reward distributions** — KTO's loss
  aversion helps.
- **Cold-start alignment** where paired data is not yet
  available.

## When NOT to use

- **When paired preferences are cheap** — DPO / IPO have stronger
  theory.
- **Very small preference datasets** — KTO's benefit needs
  ~thousands of unpaired labels.
- **When calibrated probabilities matter** — KTO's loss is not
  a proper scoring rule.

## Assumptions & caveats

- **β (KL strength)** ~ 0.1 typical; too large shrinks policy
  toward ref.
- **λ_D, λ_U (desirable / undesirable weights)** — set based on
  class imbalance.
- **Reference model** still needed (like DPO); memory doubles.
- **HALO** framework (Human-Aware Losses) generalises KTO to
  other prospect-theory value functions.

## Related in this repo

- `dpo-direct-preference-optimization`,
  `rlhf-preferences` — paired-preference cousins.
- `orpo-odds-ratio`, `simpo-simple-preference` — reference-free
  siblings.
- `constitutional-ai` — RLAIF alternative.
- `best-of-n-sampling`, `process-reward-model-prm` — inference-
  time selection with a reward model.

## Run

```
python techniques/kto-kahneman-tversky/python/kto_kahneman_tversky.py
Rscript techniques/kto-kahneman-tversky/r/kto_kahneman_tversky.R
```

**Refs:** Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D. & Kiela, D. "KTO: Model alignment as prospect theoretic optimization." *arXiv:2402.01306*, 2024; Rafailov, R. et al. "Direct preference optimization." *NeurIPS*, 2023.

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
