# 3D Gaussian Splatting (Reference §47.224)

Kerbl, Kopanas, Leimkuehler & Drettakis (2023, SIGGRAPH).
Represents a scene as **millions of 3D Gaussians** with:

    position μ_i ∈ ℝ³, covariance Σ_i (3×3 anisotropic),
    color c_i (spherical harmonics), opacity α_i ∈ [0, 1]

**Render**: project each Gaussian to 2D (splat), depth-sort,
alpha-composite. Trained end-to-end from multi-view photos via
differentiable rasterisation. **Real-time (100+ fps)** on modern
GPUs at NeRF-quality; much faster than NeRF's volumetric
ray-marching.

## Files

- `python/gaussian_splatting_3d.py` — toy splatting with 20
  Gaussians at random 3D positions + depth-sort +
  alpha-composite into a 48×48 image:
  - Rendered image with 20.7 % pixel coverage (α > 0.05).
  - Demonstrates project-to-2D + composite pipeline.
- `r/gaussian_splatting_3d.R` — recommends `graphdeco-inria/
  gaussian-splatting`, `nerfstudio` splatfacto, `gsplat`.

## When to use

- **Real-time novel view synthesis** — VR / AR, interactive tools.
- **When training compute is fast enough** — 30-60 min per scene
  on a single GPU.
- **Dynamic scenes** — 4D Gaussian Splatting extensions.

## When NOT to use

- **Non-photorealistic geometry** — 3DGS is a rendering primitive,
  not a mesh.
- **Very large scenes** without hierarchical LOD.
- **Sparse-view inputs** — needs 50-200 photos typical.

## Assumptions & caveats

- **Densification / pruning** during training — start with SfM
  points, split high-error Gaussians, prune low-opacity.
- **Spherical harmonics** for view-dependent color.
- **Anisotropic covariance** essential — isotropic Gaussians are
  blurry.
- **Alpha compositing** front-to-back with saturation threshold.

## Related in this repo

- `nerf-neural-radiance-fields` — implicit volumetric alternative.
- `video-diffusion` — related 4D scene generation.
- `sam-segment-anything` — often used for scene segmentation
  before splatting.

## Run

```
python techniques/gaussian-splatting-3d/python/gaussian_splatting_3d.py
Rscript techniques/gaussian-splatting-3d/r/gaussian_splatting_3d.R
```

**Refs:** Kerbl, B., Kopanas, G., Leimkuehler, T. & Drettakis, G. "3D Gaussian Splatting for real-time radiance field rendering." *SIGGRAPH*, 2023.

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
