# FGSM Adversarial Attack (Reference Ch 30 Robustness)

**Fast Gradient Sign Method** — the cheapest first-order white-box
adversarial attack (Goodfellow, Shlens & Szegedy 2014). One backprop step;
one sign quantisation; done.

## Formula

Under an `L_∞` budget `‖δ‖_∞ ≤ ε`:

```
x_adv  =  x  +  ε · sign( ∇_x L( f(x), y ) )
```

- Loss is the training loss on the *correct* label `y`; sign of the
  gradient pushes each coordinate the direction that increases loss.
- For a **targeted** attack toward class `y_t`, flip the sign:
  `x_adv = x − ε · sign(∇_x L(f(x), y_t))`.

## Why it works

Neural nets behave **almost linearly** in small `x`-neighbourhoods
(Goodfellow's linearity hypothesis). The linear worst-case perturbation
under an `L_∞` ball is exactly `ε · sign(∇)`. Even undefended linear
classifiers are attackable — the demo here shows it on plain logistic
regression.

## When to use

- **Sanity check** the robustness of any model — if FGSM breaks it, the
  model has no adversarial robustness at all.
- **Fast attack for training** — one-step FGSM adversarial training
  (Goodfellow 2014) as the cheapest defence.
- **Baseline** for stronger attacks (PGD, CW, DeepFool, AutoAttack).

## When NOT to use

- **As an audit of defences** — FGSM is easily circumvented by
  gradient masking. Use PGD / AutoAttack for defence evaluation.
- **Non-`L_∞` threat models** — for `L_2` or `L_0` attacks use PGD-`L_2`
  or CW.

## Files

- `python/fgsm_adversarial.py` — from-scratch FGSM on a logistic-
  regression classifier (5-d Gaussian blobs). Table of `L_∞` budget vs
  post-attack accuracy: clean 0.914 → eps=0.10 gives 0.846 → eps=0.50
  gives 0.380.
- `r/fgsm_adversarial.R` — `reticulate` + `foolbox` / `cleverhans` /
  `advertorch` / `torchattacks`.

## Assumptions & caveats

- **White-box** — attacker knows the model weights and the loss. Black-
  box variants (transfer, SPSA) exist.
- **`L_∞` ball only** — FGSM is defined for `L_∞`; other norms need
  gradient rescaling (`ε · ∇/‖∇‖₂` for `L_2`).
- **Gradient masking** by defences can *appear* to defeat FGSM without
  fixing robustness — verify with PGD.
- **Single step** — often 1.5–5× weaker than multi-step PGD; use
  multi-step for defence audits.
- **Clipping** — enforce the valid input range (e.g. `[0, 1]` for images)
  after the sign step.

## Related in this repo

- `pgd-adversarial-training`, `trades-adversarial` — multi-step attack
  + adversarial training defences.
- `randomized-smoothing` — certified defence with a coverage guarantee.
- `feature-squeezing` — cheap input-preprocessing defence.
- `label-smoothing`, `mixup`, `cutmix` — regularisers that give modest
  adversarial robustness gains.
- `deep-mlp-backprop` — the underlying training loop.

## Run

```
python techniques/fgsm-adversarial/python/fgsm_adversarial.py
Rscript techniques/fgsm-adversarial/r/fgsm_adversarial.R
```

**Refs:** Goodfellow, I.J., Shlens, J. & Szegedy, C. "Explaining and harnessing adversarial examples." *ICLR*, 2015; Szegedy, C. et al. "Intriguing properties of neural networks." *ICLR*, 2014.

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
