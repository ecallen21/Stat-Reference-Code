# Triggered analysis + conditional metrics (Reference Sec 44.13)
# Native R via stats subset + survey; Python scipy + custom.
# Run with:  Rscript triggered_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::t.test / lm on triggered subset\n")
  cat("  survey::svyglm                    -- domain estimation with correct SEs\n")
  cat("Python:\n")
  cat("  scipy.stats + custom              -- triggered subset analysis\n")
  cat("  statsmodels                       -- lm on triggered subset\n")
  cat("  causalml                          -- CACE / IV alternatives\n")
  cat("Refs: Deng & Shi (2016) 'Data-driven metric development for online controlled\n")
  cat("      experiments', KDD.\n")
}
