# Straight-Through Estimator (Reference §47.316)

Bengio, Léonard & Courville (2013). Backpropagate through a
NON-DIFFERENTIABLE step (argmax, sign, binarisation) by using
the IDENTITY gradient in the backward pass:

```
forward:  y = f(x)                      (step, argmax, sign)
backward: dL/dx = dL/dy                 (as if f = identity)
```

Biased but low-variance; workhorse for quantised networks,
VQ-VAE, and stochastic binary neurons.

## Files

- `python/straight_through_estimator.py` — Fit a
  binary-weight linear model to reproduce a target scalar
  via SGD through the sign() step. Loss drops from ~1.5 to
  0 in ~10 steps as the continuous latent w flips signs to
  match the true binary weight pattern.
- `r/straight_through_estimator.R` — `torch` (R) with
  custom autograd_function (R); `torch.autograd.Function`
  with identity backward, `torch.ao.quantization`,
  `vector_quantize_pytorch`, from-scratch (Python).

## When to use

- **Weight / activation quantisation** — training INT8 / INT4
  networks.
- **VQ-VAE codebook** — pass gradient through the nearest-
  neighbour lookup.
- **Binary / stochastic neurons** — Bengio's original use
  case.

## When NOT to use

- **When Gumbel-Softmax works** — that's smoother and less
  biased.
- **When forward pass is smooth** — no need for STE.
- **Very deep quantised networks** — the bias can accumulate;
  use learned step-size (LSQ) or PACT.

## Assumptions & caveats

- **Bias vs variance** — STE is biased; REINFORCE is
  unbiased but noisier.
- **Clipping** — often combined with `clip(x, -1, 1)` before
  sign() to bound gradient signal.
- **Learned scale factors** — INT quantisation adds scale
  parameters trained through STE.
- **Sanity check** — compare against unquantised baseline
  and against Gumbel-Softmax on the same task.

## Related in this repo

- `gumbel-softmax-relaxation` — smoother alternative for
  categorical.
- `quantization-pruning`, `awq-quantization`, `gptq-quantization`,
  `kv-cache-quantization` — quantisation methods that use STE.
- `knowledge-distillation`, `mixed-precision-training` —
  companion compression techniques.

## Run

```
python techniques/straight-through-estimator/python/straight_through_estimator.py
Rscript techniques/straight-through-estimator/r/straight_through_estimator.R
```

**Refs:** Bengio, Y., Léonard, N. and Courville, A. "Estimating or propagating gradients through stochastic neurons for conditional computation." *arXiv:1308.3432*, 2013.

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
