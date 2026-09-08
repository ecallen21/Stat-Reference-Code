# Grad-CAM Saliency (Reference §47.114)

Selvaraju, Cogswell, Das, Vedantam, Parikh & Batra (2017).
Class-discriminative visual explanation for CNNs:

    α_k^c = (1/Z) Σ_{i,j} ∂y^c / ∂A_ij^k     (per-channel weight)
    L_Grad-CAM^c(x) = ReLU( Σ_k α_k^c A^k ).

Uses ONLY the last conv-layer activations and gradients — no
model retraining, works on any CNN with any target class.

## Files

- `python/grad_cam_saliency.py` — linear-model analogue of
  Grad-CAM (real Grad-CAM needs a DL framework); demonstrates the
  ReLU-of-weighted-activation core with an 8×8 toy image whose
  class-c signal sits in the top-left 3×3 block. Grad-CAM peak
  lands on (0, 2) inside the true region.
- `r/grad_cam_saliency.R` — no first-class R port; Python
  `pytorch-grad-cam`, `captum`, `keras-vis`.

## When to use

- **CNN image classifiers** — quick visual sanity check on where the
  network is looking.
- **Model debugging** — spot spurious background reliance.
- **Regulatory / clinical documentation** — attach heatmaps to
  predictions.
- **Transformer variants**: Grad-CAM++ / Score-CAM extend to ViT
  patches.

## When NOT to use

- **Non-vision models** — use IG, SHAP, LIME instead.
- **When you need SIGNED attributions** — Grad-CAM's ReLU discards
  negative gradients (see Grad-CAM++ for improvements).
- **Very small feature maps** — heatmap upsampled to input is
  blocky.

## Assumptions & caveats

- **Requires a differentiable output-per-class**.
- **Layer choice** matters — last conv layer conventional; earlier
  layers give finer / less semantic maps.
- **Bilinear upsampling** to input resolution is the standard;
  Grad-CAM is inherently a coarse localizer.
- **Not causal** — high heatmap value ≠ necessary for the prediction;
  see fidelity-based scores (deletion / insertion).

## Related in this repo

- `smoothgrad-saliency`, `integrated-gradients`, `shap-values`,
  `shap-interactions`, `lime-local-explanations`,
  `anchor-explanations`, `counterfactual-explanations`,
  `attribution-stability` — XAI toolkit.
- `pdp-ice-plots`, `ale-accumulated-local-effects`,
  `cate-clustering-ice`, `friedmans-h-statistic` — model-agnostic
  interpretability cousins.
- `transformer-encoder`, `vision-transformer`,
  `attention-mechanism` — architectures Grad-CAM extends to.

## Run

```
python techniques/grad-cam-saliency/python/grad_cam_saliency.py
Rscript techniques/grad-cam-saliency/r/grad_cam_saliency.R
```

**Refs:** Selvaraju, R.R. et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV*, 2017; Chattopadhay, A. et al. "Grad-CAM++: Improved visual explanations for deep convolutional networks." *WACV*, 2018.

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
