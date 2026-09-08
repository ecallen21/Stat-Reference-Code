# Weighted Quantile Sum regression (Reference Sec 47.62)
# Native R via gWQS; Python via wqspy / from-scratch.
# Run with:  Rscript weighted_quantile_sum_wqs.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gWQS::gwqs                 -- Carrico et al 2015 WQS with all bells\n")
  cat("  qgcomp::qgcomp             -- Keil et al 2020 modern alternative\n")
  cat("  bkmr                       -- Bayesian kernel machine regression\n")
  cat("Python:\n")
  cat("  wqspy                      -- Python port of gWQS\n")
  cat("  from-scratch               -- see weighted_quantile_sum_wqs.py\n")
  cat("Refs: Carrico, Gennings, Wheeler & Factor-Litvak (2015) J Agric Biol\n")
  cat("      Environ Stat 20(1); Keil et al (2020) 'A quantile-based\n")
  cat("      g-computation approach to addressing the effects of exposure\n")
  cat("      mixtures', Environ Health Perspect 128(4).\n")
}
