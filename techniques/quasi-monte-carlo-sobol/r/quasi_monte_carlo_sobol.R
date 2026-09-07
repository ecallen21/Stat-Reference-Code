# Quasi-Monte Carlo -- Sobol / Halton (Reference Sec 45.9)
# Native R via randtoolbox / qrng; Python via scipy.stats.qmc / SALib.
# Run with:  Rscript quasi_monte_carlo_sobol.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  randtoolbox::sobol / halton / torus  -- low-discrepancy sequences\n")
  cat("  qrng                                   -- Owen-scrambled QMC engines\n")
  cat("  fOptions                                -- Sobol / Halton for finance\n")
  cat("  spacefillr                              -- Owen-scrambled Sobol / Halton\n")
  cat("Python:\n")
  cat("  scipy.stats.qmc.Sobol / Halton / LatinHypercube -- reference implementation\n")
  cat("  SALib                                   -- sensitivity analysis over QMC\n")
  cat("  torch.quasirandom.SobolEngine           -- GPU-side Sobol\n")
  cat("Refs: Niederreiter, H. (1992) Random Number Generation and Quasi-Monte\n")
  cat("      Carlo Methods, SIAM; Sobol, I.M. (1967) 'On the distribution of\n")
  cat("      points in a cube and the approximate evaluation of integrals',\n")
  cat("      USSR Comp Math Math Phys 7; Owen, A. (1997) 'Scrambled net variance\n")
  cat("      for integrals of smooth functions', Ann Stat 25.\n")
}
