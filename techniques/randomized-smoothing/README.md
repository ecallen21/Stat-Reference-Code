# Randomised Smoothing (Reference Ch 30 Robustness)

**Certified L2-robustness** for any base classifier by smoothing with
Gaussian noise. Cohen, Rosenfeld & Kolter (2019) turned a heuristic
"noise then vote" idea into a *proof* of adversarial robustness.

## Smoothed classifier

Given any base classifier `f` and noise level `σ`:

```
g(x)  =  argmax_c   ℙ_{δ ~ N(0, σ² I)} [ f(x + δ) = c ]
```

## Cohen's certification (Theorem 1)

Let `p_A` be the smoothed probability of the top class and `p_B` the
runner-up. For any perturbation with `‖δ‖₂ ≤ R`,

```
R  =  σ / 2 · ( Φ⁻¹(p_A) − Φ⁻¹(p_B) )
```

`g(x + δ)` is *guaranteed* to still return the top class. `Φ⁻¹` is the
standard-normal quantile function.

**In practice** we cannot compute `p_A` exactly, only estimate it from
`n` Monte-Carlo samples. Cohen replaces `(p_A, p_B)` with a lower Clopper-
Pearson confidence bound on `p_A` and `1 − lower(p_A)` as an upper on
`p_B`, giving a rigorous — but conservative — certified radius.

## Algorithm

```
1. Draw n noisy copies x + δ_i, δ_i ~ N(0, σ² I).
2. Vote: c_hat = argmax counts; k = count of c_hat.
3. p_A_low = Beta.ppf(α, k, n − k + 1)      (Clopper-Pearson lower)
4. If p_A_low ≤ 0.5 -> ABSTAIN.
   Else certify R = σ · Φ⁻¹(p_A_low).
```

## When to use

- **You need a *proof*, not an empirical audit** — self-driving, safety-
  critical medical AI, defence.
- **Compatible with any base classifier**, especially when the base was
  trained with Gaussian noise augmentation (matches deployment noise).
- **L2 threat model** — for L_∞ use Salman 2019's L2→L_∞ conversion or
  interval-bound propagation.

## When NOT to use

- **Requires large `n`** (typically 10⁴–10⁵ samples per input for
  meaningful CIs) — much slower at inference than the base classifier.
- **σ is a hard trade-off** — small σ ⇒ small certified radius; large σ
  ⇒ base accuracy under noise collapses.
- **Abstentions** on borderline inputs — the guarantee is
  correct-or-abstain, not correct-always.

## Files

- `python/randomized_smoothing.py` — from-scratch smoothed classifier
  with Clopper-Pearson lower bound + Cohen's Φ⁻¹ certified radius.
  Demo on 8-d two-Gaussian-blob problem: **σ = 0.25 → mean R ≈ 0.62;
  σ = 0.50 → 1.21; σ = 1.00 → 1.58** with 20/20 certified predictions.
- `r/randomized_smoothing.R` — `reticulate` + `smoothing-cohen` reference
  Python repo; native R via `DescTools::BinomCI` for the CI step.

## Assumptions & caveats

- **Gaussian noise → L2 certificate** only. L_∞ radii need conversion
  and are typically 10-100× smaller than the L2 radius.
- **Base classifier must be trained with noise** — a clean-trained
  ResNet collapses under σ = 0.5 augmentation; retrain with matching
  noise (Salman 2019 combines with adversarial training).
- **`n` is the accuracy vs runtime knob** — n = 100k is standard for
  ImageNet-scale certifications.
- **Confidence α** — the certified radius is w.p. `1 − α`; standard
  is `α = 10⁻³`.
- **Non-Gaussian smoothing** exists (Uniform → L_∞, Laplace → L1)
  but Gaussian is by far the most-studied.

## Related in this repo

- `fgsm-adversarial`, `pgd-adversarial-training`, `trades-adversarial`
  — *empirical* defences (no certificate).
- `noise-injection` (via `dropout-batchnorm`) — training-time analogue.
- `conformal-classification` — an orthogonal certification (over the
  data distribution, not over adversarial neighbourhoods).

## Run

```
python techniques/randomized-smoothing/python/randomized_smoothing.py
Rscript techniques/randomized-smoothing/r/randomized_smoothing.R
```

**Refs:** Cohen, J., Rosenfeld, E. & Kolter, Z. "Certified adversarial robustness via randomised smoothing." *ICML*, 2019; Salman, H. et al. "Provably robust deep learning via adversarially trained smoothed classifiers." *NeurIPS*, 2019; Lecuyer, M. et al. "Certified robustness to adversarial examples with differential privacy." *IEEE S&P*, 2019.

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
