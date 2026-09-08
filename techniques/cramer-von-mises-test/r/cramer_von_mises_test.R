# Cramer-von Mises test (Reference Sec 47.102)
# Native R via goftest / dgof; Python via scipy.stats.
# Run with:  Rscript cramer_von_mises_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  goftest::cvm.test           -- one-sample CVM against any parametric family\n")
  cat("  dgof::cvm.test              -- discrete-support variant\n")
  cat("  CvM2SL2Test / twosamples    -- two-sample CVM / permutation tests\n")
  cat("Python:\n")
  cat("  scipy.stats.cramervonmises        -- one-sample\n")
  cat("  scipy.stats.cramervonmises_2samp  -- two-sample\n")
  cat("  from-scratch                      -- see cramer_von_mises_test.py\n")
  cat("Refs: Cramer (1928); von Mises (1931); Anderson (1962) Ann Math Stat 33.\n")
}
