# Minimum Description Length (Reference Sec 34.8)
# Native R via stats + custom; Python via statsmodels.
# Run with:  Rscript minimum_description_length.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::BIC                    -- 2-part MDL asymptotic\n")
  cat("  minMDL                        -- Grunwald reference implementation (limited)\n")
  cat("Python:\n")
  cat("  scipy.optimize + custom NML  -- implement Normalised Maximum Likelihood\n")
  cat("  statsmodels                   -- BIC available on any fitted model\n")
  cat("Refs: Rissanen, J. (1978) 'Modeling by shortest data description', Automatica;\n")
  cat("      Rissanen, J. (1996) 'Fisher information and stochastic complexity',\n")
  cat("      IEEE Trans IT; Grunwald, P.D. (2007) 'The Minimum Description Length\n")
  cat("      Principle', MIT Press.\n")
}
