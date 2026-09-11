# ORPO — Odds Ratio Preference Optimization (Reference §47.198)

Hong, Lee & Thorne (2024, EMNLP). Combines SFT and preference
learning in ONE loss with **no reference model**:

    L_ORPO = L_SFT(chosen) + λ · L_OR
    L_OR   = −log σ( log-odds(chosen) − log-odds(rejected) )
    odds(y|x) = P(y|x) / (1 − P(y|x))

Since ORPO doesn't need π_ref, it halves memory vs DPO and can
train from scratch (SFT + preference simultaneously).

## Files

- `python/orpo_odds_ratio.py` — toy ORPO on 6 preference pairs
  (response 2 always chosen over other candidates):
  - Learned policy: **response 2 → 0.989** probability mass.
  - Loss trace: 2.20 → 0.01 in 150 iterations.
- `r/orpo_odds_ratio.R` — no R port; recommends `trl.ORPOTrainer`,
  `axolotl` ORPO configs.

## When to use

- **Combined SFT + preference tuning** in one pass — halves
  training time vs SFT-then-DPO.
- **Memory-tight setups** — no reference model to hold.
- **Small-to-medium models** — the paper's evaluation focus.

## When NOT to use

- **When SFT and preference stages must be decoupled** for
  ablation.
- **Very large models** — memory savings less critical than
  training-stability trade-offs.
- **When DPO / IPO / SimPO already work well** for your use case.

## Assumptions & caveats

- **λ (OR weight)** — 0.1-1.0 typical; balances SFT vs
  preference.
- **Log-odds** requires probabilities in (0, 1); clip / smooth in
  practice.
- **No KL constraint** — policy can drift far from init; add
  weight decay or LoRA rank to bound.
- **Length bias** — like DPO, ORPO benefits from length-
  normalisation.

## Related in this repo

- `dpo-direct-preference-optimization`,
  `kto-kahneman-tversky`, `simpo-simple-preference` — sibling
  preference-optimisation methods.
- `rlhf-preferences` — the RLHF pipeline ORPO short-circuits.
- `instruction-tuning-flan` — upstream SFT step ORPO folds in.
- `constitutional-ai` — RLAIF alternative.

## Run

```
python techniques/orpo-odds-ratio/python/orpo_odds_ratio.py
Rscript techniques/orpo-odds-ratio/r/orpo_odds_ratio.R
```

**Refs:** Hong, J., Lee, N. & Thorne, J. "ORPO: Monolithic preference optimization without reference model." *EMNLP*, 2024.

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
