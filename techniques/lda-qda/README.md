# Linear & Quadratic Discriminant Analysis (Reference §9.30)

Both fit a class-conditional multivariate Gaussian to each class and classify via **Bayes' rule** using estimated class priors `π_k`:

```
P(class = k | x)  ∝  π_k · N(x | μ_k, Σ_k)
```

## LDA vs. QDA

| | LDA | QDA |
|---|---|---|
| Covariance assumption | `Σ_k = Σ` (pooled) | Each class has its own `Σ_k` |
| Decision boundary | **Linear** in x | **Quadratic** in x |
| Parameters per class | `p` (mean) | `p + p(p+1)/2` (mean + Σ) |
| Data needed per class | Less | More (to estimate Σ_k) |
| Robustness | More robust when covariances are similar | More flexible when they aren't |

## Decision rules

```
LDA:  δ_k(x)  =  x' Σ⁻¹ μ_k  −  ½ μ_k' Σ⁻¹ μ_k  +  log π_k

QDA:  δ_k(x)  =  −½ log|Σ_k|  −  ½ (x−μ_k)' Σ_k⁻¹ (x−μ_k)  +  log π_k
```

Classify to `argmax_k δ_k(x)`. Posteriors come from softmax over `δ_k`.

## Files

- `python/lda_qda.py` — from-scratch fit + predict for both. Priors, means, and train accuracy match `sklearn.discriminant_analysis.{Linear,Quadratic}DiscriminantAnalysis` to 12 dp.
- `r/lda_qda.R` — from-scratch + `MASS::lda` / `MASS::qda`.

## Assumptions

- **Class-conditional multivariate normality** of features.
- **LDA additionally**: equal class covariance matrices (test via Box's M).
- Standardize features if scales differ wildly.
- Sample size per class ≥ `p` for LDA; ≥ 5p for QDA is a common rule of thumb.

## Run

```
python techniques/lda-qda/python/lda_qda.py
Rscript techniques/lda-qda/r/lda_qda.R
```

**Refs:** Fisher, R.A. "The use of multiple measurements in taxonomic problems." *Ann. Eugenics* 7(2), 179–188, 1936; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 4.3); McLachlan, G.J. *Discriminant Analysis and Statistical Pattern Recognition*, Wiley, 2004.

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
