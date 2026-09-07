# Nadaraya-Watson kernel regression (Reference Sec 5.15)
# Native R via stats::ksmooth / KernSmooth / np; Python via statsmodels.
# Run with:  Rscript nadaraya_watson_kernel_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::ksmooth                  -- simple N-W with box / normal kernels\n")
  cat("  KernSmooth::locpoly             -- local polynomial (N-W is degree 0)\n")
  cat("  np::npreg                       -- Racine's nonparametric regression\n")
  cat("  sm::sm.regression                -- Bowman-Azzalini smoothing\n")
  cat("Python:\n")
  cat("  statsmodels.nonparametric.KernelReg -- multi-dim N-W with CV bandwidth\n")
  cat("  scikit-learn KernelRidge (different: penalised, RKHS)\n")
  cat("Refs: Nadaraya, E.A. (1964) 'On estimating regression', Theor Probab Appl\n")
  cat("      9(1): 141-142; Watson, G.S. (1964) 'Smooth regression analysis',\n")
  cat("      Sankhya 26(4): 359-372; Wand & Jones (1995) Kernel Smoothing, CRC.\n")
}
