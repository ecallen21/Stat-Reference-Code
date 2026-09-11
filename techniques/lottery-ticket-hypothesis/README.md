# Lottery Ticket Hypothesis (Reference §47.161)

Frankle & Carbin (2019). Iterative magnitude pruning finds a sparse
subnetwork ("winning ticket") that, when **reset to its original
random init**, trains to dense-net accuracy in equal or fewer
iterations:

    1. Init θ_0; train dense network to θ_T.
    2. Prune p % of smallest-magnitude weights → mask m.
    3. Reset unpruned weights to their θ_0 values.
    4. Retrain masked network from θ_0.
    5. Repeat 2–4 for target sparsity.

Winning tickets exist only at their **original** init; random
reinit does not work, suggesting the init encodes structural
inductive bias for the task.

## Files

- `python/lottery_ticket_hypothesis.py` — iterative magnitude
  pruning on a sparse linear regression (n = 400, d = 100, only
  5 features are truly relevant):
  - Dense: MSE 0.073.
  - **Round 4 sparsity 94 %** (6 of 100 weights kept): MSE 0.090
    — recovers all 5 true features and nearly matches the dense
    net's MSE.
- `r/lottery_ticket_hypothesis.R` — no native R port; recommends
  `torch.nn.utils.prune`, OpenLTH (Frankle's reference impl).

## When to use

- **Model compression** — deploy a much smaller, sparser network
  with negligible accuracy loss.
- **Understanding init** — LTH shows how much of "training" is
  really finding the right subnetwork.
- **Warm-starting sparse-training research** (rigging the lottery,
  synaptic flow).

## When NOT to use

- **When you cannot re-train** — LTH always requires retraining
  the pruned sub-net from the original init.
- **Extreme sparsity (> 99 %)** — accuracy typically collapses;
  see rigging-the-lottery, sparse-training-from-scratch.
- **Very large models** — the paper's original recipe (iterative
  pruning) is expensive; use one-shot / early-bird tickets.

## Assumptions & caveats

- **Init matters** — the winning ticket only wins from θ_0;
  random reinit erases most of the advantage on non-convex nets.
- **Late resetting** (Frankle-Dziugaite-Roy-Carbin 2020): reset
  to θ_k rather than θ_0 makes tickets robust for larger
  models.
- **Global vs layer-wise** magnitude pruning — global often better.
- **Compute cost** — iterative pruning is ~k × cost of a single
  dense training run.

## Related in this repo

- `quantization-pruning` — sibling model-compression sibling.
- `neural-tangent-kernel` — theoretical framework for
  understanding wide-net training.
- `double-descent` — over-parametrised generalisation companion.
- `stochastic-weight-averaging-swa`,
  `sam-sharpness-aware-minimization` — flat-minima cousins.

## Run

```
python techniques/lottery-ticket-hypothesis/python/lottery_ticket_hypothesis.py
Rscript techniques/lottery-ticket-hypothesis/r/lottery_ticket_hypothesis.R
```

**Refs:** Frankle, J. & Carbin, M. "The lottery ticket hypothesis: Finding sparse, trainable neural networks." *ICLR*, 2019; Frankle, J., Dziugaite, G. K., Roy, D. M. & Carbin, M. "Linear mode connectivity and the lottery ticket hypothesis." *ICML*, 2020; Renda, A., Frankle, J. & Carbin, M. "Comparing rewinding and fine-tuning in neural network pruning." *ICLR*, 2020.

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
