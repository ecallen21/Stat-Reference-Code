# Inhomogeneous Poisson process (Reference Sec 47.96)
# Native R via spatstat / PtProcess / NHPoisson; Python via tick.
# Run with:  Rscript poisson_point_process_inhomog.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spatstat::rpoispp / ppm       -- inhomogeneous Poisson simulation + fit\n")
  cat("  PtProcess::mpp                 -- MLE via Nelder-Mead for parametric lambda\n")
  cat("  NHPoisson::fitPP               -- non-homogeneous Poisson process MLE\n")
  cat("Python:\n")
  cat("  tick.hawkes.SimuInhomogeneousPoisson  -- simulator\n")
  cat("  lifelines.PiecewiseExponential        -- piecewise-constant lambda MLE\n")
  cat("  scipy + custom                        -- Ogata thinning + Nelder-Mead\n")
  cat("  from-scratch                          -- see poisson_point_process_inhomog.py\n")
  cat("Refs: Cox (1955) JRSS-B 17(2); Ogata (1981) IEEE TIT 27(1); Daley &\n")
  cat("      Vere-Jones (2003) 'An Introduction to the Theory of Point Processes'.\n")
}
