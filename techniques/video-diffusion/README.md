# Video Diffusion (Reference §47.221)

Ho et al. (2022, Video Diffusion Models); Blattmann et al. (2023,
Stable Video Diffusion). Extends image diffusion to a sequence of
frames with a **3D UNet + temporal attention** layers that ensure
temporal coherence:

    x_t ∈ ℝ^{T × C × H × W}       (T noisy frames)
    UNet:  spatial attn + temporal attn (across T dim)
    loss:  ‖ε_θ(x_t, t) − ε‖²

Cascaded: low-res base → spatial upsampler → temporal
interpolator. Open models: SVD, CogVideoX, Zeroscope generate
2-6 s clips.

## Files

- `python/video_diffusion.py` — toy diffusion + temporal attention
  on 10-frame R³² sequences:
  - Original noisy: frame-to-frame variation 16.6.
  - Denoised, no temporal: 1.90.
  - **Denoised + temporal**: 0.93 (smoother, closer to ground
    truth 1.87).
- `r/video_diffusion.R` — no R port; recommends
  `diffusers.StableVideoDiffusionPipeline`, CogVideoX.

## When to use

- **Text-to-video / image-to-video** generation.
- **Video editing** with structural conditioning (depth, pose).
- **When temporal coherence matters** — plain per-frame diffusion
  flickers.

## When NOT to use

- **Real-time / low-latency** — even SVD needs seconds per clip on
  GPU.
- **Long videos** — most models limited to 2-16 s.
- **Non-visual video** (data streams) — misapplied domain.

## Assumptions & caveats

- **Temporal attention** vs 3D convolutions — attention scales
  better with T.
- **Latent-space diffusion** (SVD, Sora) is standard — pixel-
  space is prohibitive.
- **Frame rate / duration** dictated by training data; can't
  arbitrarily extend at inference.
- **Compute cost** grows as O(T²) with full temporal attention;
  cascade + windowing help.

## Related in this repo

- `diffusion-model`, `latent-diffusion-ldm`,
  `ddim-implicit-diffusion`, `classifier-free-guidance`,
  `consistency-models`, `rectified-flow` — diffusion neighbours.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder` — attention neighbours.
- `nerf-neural-radiance-fields`,
  `gaussian-splatting-3d` — 4D scene generation cousins.

## Run

```
python techniques/video-diffusion/python/video_diffusion.py
Rscript techniques/video-diffusion/r/video_diffusion.R
```

**Refs:** Ho, J. et al. "Video diffusion models." *NeurIPS*, 2022; Blattmann, A. et al. "Stable video diffusion: Scaling latent video diffusion models to large datasets." *arXiv:2311.15127*, 2023.

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
