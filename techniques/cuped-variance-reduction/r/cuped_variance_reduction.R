# CUPED variance reduction (Reference Sec 44.3)
# Native R via stats::lm (regression adjustment); Python statsmodels + custom.
# Run with:  Rscript cuped_variance_reduction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::lm                        -- OLS regression adjustment (CUPED)\n")
  cat("  sandwich                          -- robust SEs after CUPED\n")
  cat("Python:\n")
  cat("  statsmodels::OLS                  -- OLS with pre-period covariates\n")
  cat("  scipy.stats + custom\n")
  cat("Refs: Deng, Xu, Kohavi & Walker (2013) 'Improving the sensitivity of online\n")
  cat("      controlled experiments by utilizing pre-experiment data', WSDM;\n")
  cat("      Poyarkov, Drutsa, Khalyavin, Gusev & Serdyukov (2016) 'Bootstrapped and\n")
  cat("      stratified approaches in A/B testing', SIGIR.\n")
}
