# Feature Squeezing (Reference Ch 30 Robustness)

**Cheap input-preprocessing defence + detection** for image
classifiers. Xu, Evans & Qi (2018) proposed two "squeezers":

## Bit-depth reduction

Quantise each pixel from 8 bits to `k` bits:

```
s_bd(x)  =  round( x · (2^k − 1) ) / (2^k − 1)
```

Adversarial perturbations at sub-pixel levels are wiped out.

## Median spatial filter

Local-median smoothing with a `size × size` window (typical `3` or `5`):

```
s_med(x)_{i,j}  =  median( { x_{i+u, j+v} : (u, v) ∈ window } )
```

Destroys high-frequency single-pixel adversarial noise.

## Use as DEFENCE

Run the classifier on `s(x)` instead of `x`. Trades a small clean-
accuracy loss for a modest robustness gain against `L_∞` and `L_2`
attacks.

## Use as DETECTION

For any input `x`, compare the predictions before and after squeezing:

```
score(x)  =  ‖ f(x) − f(s(x)) ‖_1
```

Large `score(x)` ⇒ likely adversarial. In the paper they combine
several squeezers with an `L_1`-max over their prediction disagreements
and threshold at a validation-tuned level.

## When to use

- **Deployed classifier that already accepts image inputs** — squeezing
  is a preprocessing hook.
- **Low-compute defence** — no retraining, no gradient computation.
- **Detection pipeline** — flag suspicious inputs for human review.

## When NOT to use

- **Adversary is aware of the squeezer** — adaptive attacks (BPDA,
  He 2017) can bypass squeezing.
- **High-frequency signal matters** for the task (medical imaging fine
  detail) — squeezing may destroy signal.
- **Non-image data** — feature squeezing is specific to spatially
  structured pixel inputs.

## Files

- `python/feature_squeezing.py` — from-scratch bit-depth reduction +
  3×3 / 5×5 median filter, applied to a hand-crafted 6×6 "H"
  classifier plus an FGSM-style perturbation. Shows:
  1. Bit-depth `k = 2` partially **reverses** the adversarial drop by
     snapping perturbed pixels back to 0 or 1.
  2. Median filter size 5 produces the largest DETECTION signal
     (`|f(x) − f(s(x))| ≈ 0.28`), well above the clean-input baseline.
- `r/feature_squeezing.R` — `reticulate` + `scipy.ndimage.median_filter`
  / `skimage` in Python; native R via `imager` / `EBImage`.

## Assumptions & caveats

- **Not adversarially robust** on its own — Athalye 2018 showed that
  gradient obfuscation from squeezing is easily bypassed by BPDA.
- **Multiple squeezers** — combining bit-depth + median + non-local
  means gives stronger detection than any single squeezer.
- **Threshold** for detection is validation-tuned; measure false-positive
  rate on clean data.
- **Colour vs greyscale** — squeeze each channel independently.
- **Interaction with augmentation** — training with random Gaussian
  noise or random JPEG compression makes the model *tolerant* to
  squeezing, sacrificing some detection signal.

## Related in this repo

- `fgsm-adversarial`, `pgd-adversarial-training`, `trades-adversarial`
  — the attacks squeezing tries to defend against.
- `randomized-smoothing` — certified defence (proof, not preprocessing).
- `mixup`, `cutmix`, `label-smoothing` — training-time regularisers.
- `convolutional-nn`, `image-classification` (if present) — the models
  squeezing is typically bolted onto.

## Run

```
python techniques/feature-squeezing/python/feature_squeezing.py
Rscript techniques/feature-squeezing/r/feature_squeezing.R
```

**Refs:** Xu, W., Evans, D. & Qi, Y. "Feature squeezing: detecting adversarial examples in deep neural networks." *NDSS*, 2018; He, W. et al. "Adversarial example defence: ensembles of weak defences are not strong." *WOOT*, 2017; Athalye, A., Carlini, N. & Wagner, D. "Obfuscated gradients give a false sense of security." *ICML*, 2018.

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
