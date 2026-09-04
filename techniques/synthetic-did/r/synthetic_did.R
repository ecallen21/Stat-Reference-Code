# Synthetic Difference-in-Differences (Reference Sec 35.10)
# Native R via synthdid; Python via port.
# Run with:  Rscript synthetic_did.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  synthdid                     -- Arkhangelsky et al. reference (Athey lab)\n")
  cat("  gsynth                        -- Generalized synthetic control (adjacent)\n")
  cat("  augsynth                      -- augmented synthetic control\n")
  cat("Python:\n")
  cat("  synthdid.py                   -- community port of the R package\n")
  cat("  pysyncon                       -- Abadie's synthetic control (adjacent)\n")
  cat("Refs: Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. & Wager, S.\n")
  cat("      (2021) 'Synthetic Difference in Differences', American Economic Review.\n")
}
