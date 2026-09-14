# Performer FAVOR+ (Reference §47.387)

Choromanski et al. (2021 ICLR). Approximate the SOFTMAX
KERNEL `exp(qᵀk / √d)` using positive orthogonal random
features `φ`:

```
E[φ(q)ᵀ φ(k)] = exp(qᵀk)
φ(x) = exp(−‖x‖² / 2) · exp(Wx) / √m,  W_i i.i.d. Gaussian
```

Then linear-cost attention:

```
attention(Q, K, V) ≈ φ(Q) · (φ(K)ᵀ V) / (φ(Q) · φ(K)ᵀ 1)
```

Complexity `O(n · m · d)` — LINEAR in `n`. FAVOR+ gives an
UNBIASED estimate of the softmax kernel (unlike Nyströmformer
which is deterministic). Used in speech, protein-folding
long-context models.

## Files

- `python/performer_random_features.py` — n=256, d=32, small-
  norm Q, K (Performer's regime; large norms make exp() blow
  up). Relative error decays with m: m=16 → 0.23, m=64 →
  0.14, m=256 → 0.08. Consistent Monte-Carlo convergence
  `O(1/√m)`.
- `r/performer_random_features.R` — reticulate to
  `performer-pytorch` (R); `performer-pytorch`,
  `x_transformers.Performer`, from-scratch (Python).

## When to use

- **Extremely long sequences** — protein folding, genomics,
  full-book language modelling, high-resolution audio.
- **When kernelised attention is a good abstraction** — the
  FAVOR+ trick generalises to any positive kernel.
- **When causal attention is essential** — Performer supports
  causal masking cheaply via prefix sums.

## When NOT to use

- **Short sequences** — variance in the FAVOR+ estimate
  dominates.
- **When you need exact softmax** — Performer approximates.
- **When Q, K have large norms** — features blow up; must
  scale down or use different kernel.

## Assumptions & caveats

- **Number of random features m** — 128-256 typical for good
  accuracy; error scales `O(1/√m)`.
- **Orthogonal random features** — Choromanski et al show
  orthogonal (via QR) reduces variance vs plain Gaussian.
- **Positive-feature stability** — original PERFORMER (2020)
  used sin/cos features which gave large negative-value
  variance; FAVOR+ (2021) fixes with the exp-features.
- **Normalisation** — attention weights should sum to 1;
  small denominator ⇒ instability, add ε to protect.
- **Training-inference gap** — FAVOR+ can be biased on
  finite m; Performer papers show it works empirically.

## Related in this repo

- `reformer-lsh-attention`, `longformer-sparse-attention`,
  `linformer-projection`, `nystromformer-approximation` —
  efficient-attention siblings.
- `random-fourier-features`, `nystrom-approximation`,
  `randomized-svd` — kernel-approximation cousins.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention` — attention
  neighbours.

## Run

```
python techniques/performer-random-features/python/performer_random_features.py
Rscript techniques/performer-random-features/r/performer_random_features.R
```

**Refs:** Choromanski, K., Likhosherstov, V., Dohan, D., et al. "Rethinking attention with Performers." In *ICLR*, 2021.

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
