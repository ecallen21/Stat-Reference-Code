# TRADES Adversarial Training (Reference Ch 30 Robustness)

**TRadeoff-inspired Adversarial DEfense via Surrogate loss** — Zhang et
al. (2019). Winner of the 2018 NeurIPS Adversarial Vision Challenge and
a Pareto-improvement over Madry's PGD-AT on the clean-vs-robust
frontier.

## Loss

```
L_TRADES  =  CE( f_θ(x), y )
          +  β · KL( f_θ(x)  ‖  f_θ(x_adv) )
```

The adversary `x_adv` is found by **PGD that maximises the KL term**
(not the CE):

```
x_adv⁽⁰⁾ = x + U(−ε, ε)
x_adv⁽ᵗ⁾ = Π_{‖·−x‖_∞ ≤ ε} ( x_adv⁽ᵗ⁻¹⁾ + α · sign( ∇_x KL( f_θ(x) ‖ f_θ(x_adv⁽ᵗ⁻¹⁾) ) ) )
```

## The `β` knob

| β    | Behaviour                                                         |
|------|-------------------------------------------------------------------|
| 0    | Pure clean CE — no robustness.                                    |
| 1–2  | Small robustness gain with negligible clean-accuracy loss.        |
| 6    | Reference paper's setting; comparable to Madry PGD-AT at 1/3 cost.|
| ∞    | Only the KL term matters; collapses toward Madry PGD-AT.          |

## Advantages over Madry (2018)

- Explicit trade-off knob `β`.
- Better clean accuracy at fixed robust accuracy on CIFAR-10.
- Backward-friendly loss — no need to run PGD to convergence.

## When to use

- **Deployment where clean accuracy still matters** but adversarial
  inputs are plausible.
- **Any place PGD-AT is used** — TRADES is usually a drop-in improvement.
- **Fine-tune the trade-off** — grid-search `β` on a validation set.

## Files

- `python/trades_adversarial.py` — from-scratch TRADES for binary
  logistic regression: KL-driven PGD inner loop + CE + β·KL outer loop.
  Demo: **clean 0.914 (no defence) vs 0.900 (TRADES β=6)**;
  **eps=0.40 PGD accuracy 0.490 → 0.554** at the higher β.
- `r/trades_adversarial.R` — `reticulate` + `trades-pytorch` reference
  repo / `advertorch` / `torchattacks`.

## Assumptions & caveats

- **Inner attack must maximise KL** — not CE — for the theory bound to
  apply. A common bug: reuse the Madry PGD helper.
- **KL is symmetric-looking but not symmetric** — the direction
  `KL(clean ‖ adv)` is what appears in the theory.
- **β schedule** — some implementations anneal `β` from 0 upward for
  stability.
- **BatchNorm dual-BN** as in Madry PGD-AT applies here too.
- **Evaluate with AutoAttack** — a defence that passes only PGD often
  has gradient masking; TRADES is generally robust to this but confirm.

## Related in this repo

- `fgsm-adversarial`, `pgd-adversarial-training` — attack + baseline
  defence.
- `randomized-smoothing` — certified defence alternative.
- `label-smoothing`, `mixup`, `cutmix` — cheap regularisers that
  stack with TRADES.
- `jacobian-regularization`, `spectral-normalization` — smoothness-
  based defence families.

## Run

```
python techniques/trades-adversarial/python/trades_adversarial.py
Rscript techniques/trades-adversarial/r/trades_adversarial.R
```

**Refs:** Zhang, H. et al. "Theoretically principled trade-off between robustness and accuracy." *ICML*, 2019; Madry, A. et al. "Towards deep learning models resistant to adversarial attacks." *ICLR*, 2018.

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
