# First-hitting-time model (Reference Sec 11.28)
# Native R via threg / SMPracticals; Python via scipy.stats.invgauss.
# Run with:  Rscript first_hitting_time_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  threg                      -- Whitmore first-hitting-time regression\n")
  cat("  flexsurv::flexsurvreg(dist='gen.gamma') -- includes IG as special case\n")
  cat("  SMPracticals + custom       -- IG survival with covariate links\n")
  cat("  survival::survreg(dist='invgauss') -- available in some builds\n")
  cat("Python:\n")
  cat("  scipy.stats.invgauss + custom censored MLE (see first_hitting_time_model.py)\n")
  cat("  lifelines InverseGaussianAFTFitter (community fork)\n")
  cat("Refs: Whitmore, G.A. (1986) 'First passage time models for duration data:\n")
  cat("      regression structures and competing risks', The Statistician 35(2);\n")
  cat("      Aalen, O.O., Borgan, O. & Gjessing, H.K. (2008) Survival and Event\n")
  cat("      History Analysis, Springer, ch 12.\n")
}
