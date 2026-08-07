# Synthetic Control Method (Reference §15.10)
# R via Synth (Abadie-Diamond-Hainmueller) or gsynth (generalized).
# Run with:  Rscript synthetic_control.R

if (sys.nframe() == 0) {
  cat("Reference implementation:\n")
  cat("  Synth::synth(dataprep(...))  -- ADH 2010 canonical package\n")
  cat("  gsynth::gsynth(...)          -- generalized SC with covariates\n")
  cat("  Synth's California-tobacco or Basque example is the standard demo.\n")
}
