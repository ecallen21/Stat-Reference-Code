# 2PL / 3PL IRT (Reference Sec 20.4, 20.5)
# Native R via ltm / mirt; Python girth / py-irt + custom.
# Run with:  Rscript irt_2pl_3pl.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ltm::ltm / rasch / gpcm           -- 1PL / 2PL / GPCM item models\n")
  cat("  mirt                                -- flexible IRT (multidim, non-dichotomous)\n")
  cat("  TAM                                 -- test analysis modules (Rasch family)\n")
  cat("Python:\n")
  cat("  girth (Item Response Theory)      -- 1PL / 2PL / 3PL / graded response\n")
  cat("  py-irt                              -- Bayesian IRT (PyStan / Turing)\n")
  cat("  custom (marginal MLE via GH quadrature)\n")
  cat("Refs: Birnbaum (1968) 'Some latent trait models', in Statistical Theories of\n")
  cat("      Mental Test Scores (Lord & Novick); Baker & Kim (2004) Item Response\n")
  cat("      Theory: Parameter Estimation Techniques, CRC.\n")
}
