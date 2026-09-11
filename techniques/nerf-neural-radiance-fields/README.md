# NeRF — Neural Radiance Fields (Reference §47.225)

Mildenhall et al. (2020, ECCV). Represents a scene as an MLP:

    F_θ : (x, y, z, θ_view, φ_view) → (RGB, σ)

**Rendering** a pixel: cast a ray, sample N points along it,
query the MLP, and volume-composite:

    C(r) = Σ_i T_i · (1 − exp(−σ_i · δ_i)) · c_i
    T_i  = exp(−Σ_{j<i} σ_j · δ_j)

Trained from multi-view photos with pixel MSE loss. Slow (hours
per scene) — largely superseded by Gaussian Splatting for
real-time work.

## Files

- `python/nerf_neural_radiance_fields.py` — full volume-rendering
  pipeline with a HAND-CRAFTED radiance field (sphere density +
  position-based color). Renders a 32×32 image via 64 ray samples:
  - **77.4 % of pixels have nonzero coverage**, mean depth 3.38.
  - Demonstrates ray-casting + volume compositing.
- `r/nerf_neural_radiance_fields.R` — recommends `nerfstudio`,
  `nvlabs/instant-ngp`.

## When to use

- **Photorealistic novel view synthesis** from multi-view photos.
- **Small / medium scenes** where quality > speed.
- **Research** on scene representations (many NeRF variants).

## When NOT to use

- **Real-time serving** — GS is much faster.
- **Reflective / view-inconsistent** surfaces — plain NeRF
  struggles (see Ref-NeRF).
- **Very sparse views** — need 30+ photos typically.

## Assumptions & caveats

- **Positional encoding** (Fourier features) is essential for
  high-frequency detail.
- **Hierarchical sampling** (coarse + fine networks) improves
  fidelity.
- **Camera pose** must be known — usually from COLMAP SfM.
- **Instant-NGP** (Müller 2022) uses hash-grid features to
  train in seconds instead of hours.

## Related in this repo

- `gaussian-splatting-3d` — faster modern alternative.
- `video-diffusion`, `latent-diffusion-ldm` — generative-scene
  cousins.
- `neural-ode` — related neural implicit representation.

## Run

```
python techniques/nerf-neural-radiance-fields/python/nerf_neural_radiance_fields.py
Rscript techniques/nerf-neural-radiance-fields/r/nerf_neural_radiance_fields.R
```

**Refs:** Mildenhall, B. et al. "NeRF: Representing scenes as neural radiance fields for view synthesis." *ECCV*, 2020; Müller, T. et al. "Instant neural graphics primitives with a multiresolution hash encoding." *SIGGRAPH*, 2022.

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
