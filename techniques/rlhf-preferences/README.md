# RLHF: Preference-Based Fine-Tuning (Reference §28.10)

Christiano et al. (2017) → Ouyang et al. (2022, InstructGPT). Align a
language model to human preferences via a **learned reward model** + RL,
or via direct-preference methods that skip the RL step.

## Standard pipeline

1. **SFT** — supervised fine-tune on human demonstrations (behavioural cloning over tokens).
2. **Reward model** — collect preference pairs `(x, y_w ≻ y_l)`; fit under Bradley-Terry:
   ```
   P(y_w ≻ y_l | x) = σ(r_θ(x, y_w) − r_θ(x, y_l))
   ```
   Maximum-likelihood on the preference dataset.
3. **RL** — optimise `π_θ` against `r_θ` with a KL penalty back to the SFT model:
   ```
   max_θ  𝔼_{x, y ~ π_θ} [ r_θ(x, y) − β · KL(π_θ || π_SFT) ]
   ```
   Usually implemented with PPO (see `ppo-clipped`).

## Direct preference methods

- **DPO** (Rafailov 2023) — closed-form loss on the log-ratio; skips both the explicit reward model and the RL stage:
  ```
  L_DPO = − log σ( β · [ (log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l)) ] )
  ```
  Under Bradley-Terry, the optimum coincides with the RLHF pipeline's optimum, at a fraction of the compute.
- **IPO** (Azar 2023) — identity-preference variant that fixes DPO's over-fitting.
- **KTO** (Ethayarajh 2024) — Kahneman-Tversky utility model; works from single-preference labels (👍 / 👎) instead of pairs.
- **ORPO** (Hong 2024) — joint one-stage SFT + odds-ratio preference.
- **GRPO** (DeepSeek 2024) — group-relative advantage; used in reasoning-training pipelines (o1, DeepSeek-R1).

## When to use

- **LLM alignment** — chat, safety, style adherence, refusal training.
- **Response quality tuning** for any generative model where humans can label preferences.
- **Reward hacking** is the perennial risk — the model learns the labeller's biases, not the "true" quality.
- **NOT for tasks with a clean reward** (exact-match QA, code exec) — direct RL on that reward is usually better.

## Files

- `python/rlhf_preferences.py` — from-scratch **Bradley-Terry reward-model MLE** and **DPO** on a linear-features toy. Demo: 200 preference pairs generated from a linear reward `θ* = [1, −1, 0.5]` with 10% label noise:
  - BT reward MLE recovers `θ ≈ [0.98, −1.03, 0.48]` — cosine to truth = 1.000.
  - DPO learns the same direction (cosine 1.000) — matches theory: DPO's optimum is proportional to the BT reward.
- `r/rlhf_preferences.R` — `reticulate` + `huggingface TRL` (`PPOTrainer, DPOTrainer, GRPOTrainer, RLOOTrainer, ORPOTrainer`), `trlx`, `OpenRLHF`, `LLaMA-Factory`, `axolotl`.

## Assumptions & caveats

- **Bradley-Terry assumes transitive preferences** — real humans are inconsistent; regularise the RM (dropout / temperature).
- **Reward-model over-optimisation** — Goodhart's law: a policy that maxes the RM starts producing outputs the RM likes but humans don't. Use KL to the SFT model + early-stopping on holdout preferences.
- **DPO vs PPO** — DPO is cheaper and often just as good on small-scale preference sets; PPO scales better on very-large preference / reward pipelines with high compute budget.
- **Preference labellers** are the ceiling — bias, boredom, ambiguity in the labelling instructions all leak into the model.
- **RLAIF** (RL from AI feedback) — use a strong LLM as the labeller for cheap scale; substitutes constitutional principles for human raters.
- **Constitutional AI** (Anthropic 2022) — self-critique + RLAIF pipeline for alignment without direct human labels on every response.

## Related in this repo

- `ppo-clipped` — the RL algorithm typically used in stage 3.
- `imitation-learning` — SFT is BC over tokens.
- `transformer-decoder`, `text-generation-decoding` — the LM policy being fine-tuned.
- `bradley-terry` — the pairwise-preference model used in stage 2.

## Run

```
python techniques/rlhf-preferences/python/rlhf_preferences.py
Rscript techniques/rlhf-preferences/r/rlhf_preferences.R
```

**Refs:** Christiano, P.F. et al. "Deep reinforcement learning from human preferences." *NeurIPS*, 2017; Ouyang, L. et al. "Training language models to follow instructions with human feedback (InstructGPT)." *NeurIPS*, 2022; Rafailov, R. et al. "Direct preference optimization: your language model is secretly a reward model (DPO)." *NeurIPS*, 2023; Ethayarajh, K. et al. "KTO: model alignment as prospect-theoretic optimization." *arXiv:2402.01306*, 2024.

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
