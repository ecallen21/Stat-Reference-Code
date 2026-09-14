# Morris Elementary Effects (Reference Sec 47.335)
# Native R via sensitivity; Python via SALib / from-scratch.
# Run with:  Rscript morris_elementary_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sensitivity::morris             -- classic Morris screening (mu*, sigma)\n")
  cat("  sensitivity::morrisMultOut      -- multi-output extension\n")
  cat("  fanovaGraph                     -- graphical extension\n")
  cat("Python:\n")
  cat("  SALib.analyze.morris\n")
  cat("  UQpy.sensitivity.MorrisSensitivity\n")
  cat("  From-scratch (see morris_elementary_effects.py)\n")
  cat("Refs: Morris, M.D. (1991) 'Factorial sampling plans for preliminary\n")
  cat("      computational experiments', Technometrics 33(2);\n")
  cat("      Campolongo, F. & Braddock, R. (1999) 'Sensitivity analysis of the\n")
  cat("      IMAGE greenhouse model', Env Model Softw 14(3).\n")
}
