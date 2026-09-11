# Reservoir Sampling (Reference Sec 47.246)
# Native R via base sample / stream / custom; Python via numpy / from-scratch.
# Run with:  Rscript reservoir_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  base::sample            -- for known-length uniform sampling\n")
  cat("  stream                  -- data stream framework with reservoir samplers\n")
  cat("  ldstatsHD::reservoir    -- reservoir helpers for high-dim streams\n")
  cat("  custom loop             -- Algorithm R and A-Res are ~5 lines\n")
  cat("Python:\n")
  cat("  numpy.random.choice (for known-length)\n")
  cat("  From-scratch generator (see reservoir_sampling.py)\n")
  cat("Refs: Vitter, J.S. (1985) 'Random Sampling with a Reservoir',\n")
  cat("      ACM Trans Math Softw 11(1); Efraimidis, P.S. & Spirakis, P.G.\n")
  cat("      (2006) 'Weighted random sampling with a reservoir', IPL 97.\n")
}
