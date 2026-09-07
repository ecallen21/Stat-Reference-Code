# Density ratio estimation (Reference Sec 46.18)
# Native R via densratio; Python via densratio / from-scratch.
# Run with:  Rscript density_ratio_estimation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  densratio                   -- Sugiyama et al. uLSIF, KLIEP, RuLSIF\n")
  cat("  KRLS / dbarts               -- alt kernel / BART approaches\n")
  cat("  probabilistically::odds     -- classifier-based ratio helpers\n")
  cat("Python:\n")
  cat("  densratio (pip)             -- Python port of Sugiyama's densratio\n")
  cat("  scikit-learn LogisticRegression + odds trick\n")
  cat("  KLIEP / uLSIF implementations in individual repos\n")
  cat("Refs: Sugiyama, M., Suzuki, T. & Kanamori, T. (2012) Density Ratio\n")
  cat("      Estimation in Machine Learning, Cambridge; Kanamori, T., Hido, S.\n")
  cat("      & Sugiyama, M. (2009) 'A least-squares approach to direct\n")
  cat("      importance estimation', JMLR 10.\n")
}
