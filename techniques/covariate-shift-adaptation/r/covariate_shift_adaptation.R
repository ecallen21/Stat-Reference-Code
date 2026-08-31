# Covariate-shift adaptation (Reference Ch 29 Uncertainty Quantification)
# Density-ratio importance weighting; R has good native packages.
# Run with:  Rscript covariate_shift_adaptation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  densratio                   -- KLIEP, uLSIF, RuLSIF; SOTA density ratio estimators\n")
  cat("  transport                   -- optimal transport (an alternative to reweighting)\n")
  cat("  DRDID, WeightIt             -- doubly-robust / balancing-weight causal estimators\n")
  cat("Python:\n")
  cat("  densratio                   -- Python port of KLIEP / uLSIF\n")
  cat("  adapt                       -- awesome-domain-adaptation toolkit (KLIEP, KMM, ...)\n")
  cat("  scikit-learn                 -- LogisticRegression for pooled classifier probability trick\n")
  cat("Refs: Shimodaira, H. (2000) 'Improving predictive inference under covariate shift by\n")
  cat("      weighting the log-likelihood function', JSPI; Sugiyama, M., Suzuki, T. & Kanamori, T.\n")
  cat("      (2012) 'Density Ratio Estimation in Machine Learning', Cambridge U.P.;\n")
  cat("      Bickel, S., Bruckner, M. & Scheffer, T. (2007) 'Discriminative learning\n")
  cat("      for differing training and test distributions', ICML.\n")
}
