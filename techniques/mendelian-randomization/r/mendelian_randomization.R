# Mendelian randomization (Reference §15.x extra)
# R via MendelianRandomization or TwoSampleMR.
# Run with:  Rscript mendelian_randomization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MendelianRandomization::mr_ivw(input)                 -- inverse-variance weighted\n")
  cat("  MendelianRandomization::mr_egger(input)               -- pleiotropy intercept\n")
  cat("  MendelianRandomization::mr_median(input, weighting)   -- simple / weighted median\n")
  cat("  MendelianRandomization::mr_mbe(input, phi)            -- weighted mode-based estimator\n")
  cat("  TwoSampleMR::mr(dat)                                   -- runs the full panel\n")
  cat("  TwoSampleMR::mr_pleiotropy_test(dat)                  -- Egger intercept test\n")
  cat("  TwoSampleMR::mr_heterogeneity(dat)                    -- Cochran Q heterogeneity\n")
}
