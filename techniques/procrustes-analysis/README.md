# Procrustes Analysis (Reference §9.16)

Given two point configurations `X` and `Y` (both `n × p`), find the orthogonal transformation (rotation + reflection), optional scale, and optional translation that best aligns `Y` with `X` in the least-squares sense:

```
minimize   ‖X − (s · Y · Q + t)‖²_F
over        s (scalar), Q (p × p orthogonal), t (row vector)
```

## Applications

- **Shape analysis** in morphometrics — comparing anatomical landmarks across specimens.
- **Alignment of coordinate maps** across labs (e.g. gene-expression PCA scores).
- **Bootstrap stability** of PCA loadings — Procrustes-align each bootstrap loading to the original before comparing.
- **Multi-view sensor alignment**.

Extends to **generalized Procrustes analysis** for aligning K > 2 configurations simultaneously (see `vegan::procrustes` and `shapes::procGPA` in R).

## Files

- `python/procrustes_analysis.py` — SVD-based ordinary Procrustes; recovers a 30° rotation and 2× scale from synthetic data (fitted s = 0.499 vs true 0.5). Cross-check against `scipy.spatial.procrustes`.
- `r/procrustes_analysis.R` — thin wrapper around `vegan::procrustes`.

## Assumptions

- Same number of points and same dimension in both configurations.
- Correspondence between points (row `i` of X ↔ row `i` of Y).
- For shape analysis, remove size and location first (Procrustes handles the rest).

## Run

```
python techniques/procrustes-analysis/python/procrustes_analysis.py
Rscript techniques/procrustes-analysis/r/procrustes_analysis.R
```

**Refs:** Gower, J.C. "Generalized Procrustes analysis." *Psychometrika* 40(1), 33–51, 1975; Dryden, I.L. & Mardia, K.V. *Statistical Shape Analysis*, 2nd ed., Wiley, 2016.

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
