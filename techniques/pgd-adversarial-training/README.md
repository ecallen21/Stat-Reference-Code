# PGD + Adversarial Training (Reference Ch 30 Robustness)

**Projected Gradient Descent** (PGD) is the multi-step generalisation of
FGSM and the standard first-order attack for evaluating adversarial
robustness. **Adversarial training** trains on PGD-generated examples to
minimise the saddle-point objective of Madry et al. (2018).

## PGD attack

Under an `L_∞` budget `ε` and step size `α`:

```
x_adv⁽⁰⁾  =  x  +  U(−ε, ε)              (random start; optional)
for t = 1..T:
    x_adv⁽ᵗ⁾ = Π_{‖·−x‖_∞ ≤ ε} ( x_adv⁽ᵗ⁻¹⁾ + α · sign( ∇_x L(f(x_adv⁽ᵗ⁻¹⁾), y) ) )
```

`Π` is the projection onto the `L_∞` ball (a per-coordinate clip).
Typical settings: `α = ε / 4`, `T = 10–40`. Random start turns PGD into
a stochastic attack and eliminates degenerate zero-gradient failures.

## Adversarial training (Madry 2018)

The **saddle-point** training objective is

```
min_θ  𝔼_{(x,y) ~ D}  [  max_{‖δ‖_∞ ≤ ε}  L( f_θ(x + δ), y )  ]
```

Approximated by generating PGD examples every SGD step and optimising the
loss on **them**, not the clean inputs. The resulting model has
substantially higher accuracy under PGD attack at the cost of a small
drop in clean accuracy.

## When to use

- **Robustness evaluation** — PGD (with random start, many steps, many
  restarts) is the gold-standard first-order audit. Combine with
  AutoAttack for a rigorous certificate.
- **Robust classification** — Madry PGD-AT is the reference defence.
  Weaker but cheaper: TRADES (`trades-adversarial`), FGSM-AT.
- **Any deployment where adversarial inputs are plausible** — spam
  detection, malware classification, content moderation, self-driving.

## When NOT to use

- **Clean-only accuracy matters** — PGD-AT usually costs 3–10 % clean
  accuracy on ImageNet-scale models.
- **Compute is scarce** — inner PGD adds a `T`-fold cost to every SGD
  step. Free adversarial training (Shafahi 2019) amortises the cost.

## Files

- `python/pgd_adversarial_training.py` — from-scratch PGD attack with
  random start + projection; Madry-style PGD-AT with `T = 7` inner
  steps. Demo on logistic regression: **clean-trained** eps=0.40
  PGD accuracy 0.490 → **PGD-trained** eps=0.40 PGD accuracy 0.560, at
  a modest clean-accuracy cost (0.914 → 0.894).
- `r/pgd_adversarial_training.R` — `reticulate` + `foolbox` /
  `cleverhans` / `advertorch` / `robustness` (Madry-lab).

## Assumptions & caveats

- **`L_∞` threat model** — swap sign for gradient / L2-norm for
  `L_2` PGD.
- **Number of restarts** — a single-restart PGD can miss the
  worst-case perturbation; standard audits use 20+ restarts.
- **Gradient masking** — defences that *appear* to break PGD often
  have gradient obfuscation, not real robustness (Athalye 2018).
  Cross-check with AutoAttack or transfer attacks.
- **Trade-off is non-monotone** — very large `ε` during training can
  collapse clean accuracy to random.
- **Batch statistics** — with BatchNorm, adversarial + clean batches
  should use *different* running stats (dual BN, Xie 2020).

## Related in this repo

- `fgsm-adversarial` — the single-step version of PGD.
- `trades-adversarial` — KL-based defence with a controllable clean/robust
  trade-off.
- `randomized-smoothing` — a certified defence (not just empirical).
- `label-smoothing`, `mixup`, `cutmix` — regularisers with a small
  robustness bonus.
- `gradient-clipping`, `spectral-normalization`, `jacobian-regularization`
  — smoothness-based defences.

## Run

```
python techniques/pgd-adversarial-training/python/pgd_adversarial_training.py
Rscript techniques/pgd-adversarial-training/r/pgd_adversarial_training.R
```

**Refs:** Madry, A. et al. "Towards deep learning models resistant to adversarial attacks." *ICLR*, 2018; Athalye, A., Carlini, N. & Wagner, D. "Obfuscated gradients give a false sense of security." *ICML*, 2018.

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
