# Sequential analysis / SPRT (Reference Sec 37.6)
# Native R via gsDesign / rpact; Python via scipy.
# Run with:  Rscript sequential_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gsDesign                     -- group-sequential designs (O'Brien-Fleming, Pocock)\n")
  cat("  rpact                         -- confirmatory adaptive clinical trials\n")
  cat("  ldbounds                      -- Lan-DeMets alpha spending\n")
  cat("  sprtt                          -- classical SPRT\n")
  cat("Python:\n")
  cat("  scipy.stats                    -- building blocks\n")
  cat("  seqtest / pyalphastate          -- niche implementations\n")
  cat("Refs: Wald, A. (1945) 'Sequential tests of statistical hypotheses',\n")
  cat("      Annals of Mathematical Statistics; Whitehead, J. (1997) 'The Design and\n")
  cat("      Analysis of Sequential Clinical Trials', Wiley.\n")
}
