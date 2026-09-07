# Score matching (Reference Sec 46.17)
# From-scratch in R and Python; deep-learning variants live in pytorch / jax.
# Run with:  Rscript score_matching.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  scoringutils                    -- proper-scoring-rule diagnostics (related)\n")
  cat("  No dedicated CRAN implementation -- roll from Hyvarinen's identity\n")
  cat("Python:\n")
  cat("  sm4mb (score matching for models with bounds)\n")
  cat("  torch + custom loss -- SGD score matching (denoising / sliced variants)\n")
  cat("  score-sde / diffusion-models    -- score-based generative code releases\n")
  cat("Refs: Hyvarinen, A. (2005) 'Estimation of non-normalized statistical models\n")
  cat("      by score matching', JMLR 6(24); Song, Y. & Ermon, S. (2019)\n")
  cat("      'Generative modeling by estimating gradients of the data\n")
  cat("      distribution', NeurIPS.\n")
}
