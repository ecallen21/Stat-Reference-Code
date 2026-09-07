# DPO -- Direct Preference Optimization (Reference §47.25)

Rafailov et al. (2023, NeurIPS). Trains a policy `π_θ` directly on
preference pairs **without** training a separate reward model or
running PPO.

## Loss

For prompt `x` with preferred `y_w` and dispreferred `y_l`:

    L = − E[ log σ( β · (log π_θ(y_w|x) / π_ref(y_w|x)
                        − log π_θ(y_l|x) / π_ref(y_l|x))) ]

Equivalent to RLHF's Bradley-Terry + KL-regularised optimum in
closed form.

## Advantages over PPO + RM

- No separate reward model.
- No PPO / rollout / value network.
- Simpler, cheaper, more stable in practice.

## Files

- `python/dpo_direct_preference_optimization.py` — manual softmax
  policy + DPO loss + gradient step from scratch on a bandit-style
  toy. Demo (5 actions with rewards 0.1, 0.3, 0.9, 0.5, 0.7,
  uniform reference): larger `β` concentrates mass on the argmax
  (β=0.5 → 74% on action 2; β=0.1 → 39%; stays closer to reference).
- `r/dpo_direct_preference_optimization.R` — no R implementations;
  describes HuggingFace TRL: `DPOTrainer`, `IPOTrainer`,
  `KTOTrainer`, and the PEFT + TRL + transformers stack (Python).

## When to use

- **Aligning LMs to human / synthetic preferences** — replacement
  for RLHF PPO in most alignment pipelines.
- **Instruction tuning after SFT** — DPO on preference pairs
  polishes the SFT-tuned model.
- **Small-team alignment budgets** — one training loop, one loss.

## When NOT to use

- **Reward-only feedback** (no pairs) — use RL / RLHF instead.
- **Multi-objective preferences** — use vector-DPO / IPO variants.
- **Very long sequences with sparse preferences** — token-level
  variants (KTO, GRPO) may be more sample-efficient.

## Assumptions & caveats

- **β (KL regulariser)** — sets how far the policy strays from
  reference; sweep β ∈ {0.05, 0.1, 0.5}.
- **Reference model** — usually the SFT model; if reference is bad,
  DPO inherits its problems.
- **Reward hacking / reward-model bias** avoided but preference-bias
  is inherited from labellers.
- **Length bias** — DPO tends to reward longer completions; use
  length-normalised DPO or IPO for correction.

## Related in this repo

- `rlhf-preferences` — the RLHF+PPO alternative.
- `lora-peft` — commonly stacked: LoRA + DPO for efficient tuning.
- `transformer-decoder`, `attention-mechanism`,
  `retrieval-augmented-generation` — LM machinery.

## Run

```
python techniques/dpo-direct-preference-optimization/python/dpo_direct_preference_optimization.py
Rscript techniques/dpo-direct-preference-optimization/r/dpo_direct_preference_optimization.R
```

**Refs:** Rafailov, R. et al. "Direct Preference Optimization: your language model is secretly a reward model." *NeurIPS*, 2023; Azar, M.G. et al. "A general theoretical paradigm to understand learning from human preferences" (IPO), 2023; Ethayarajh, K. et al. "KTO: Model alignment as prospect theoretic optimization." 2024.

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
