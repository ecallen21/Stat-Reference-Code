# Person-fit statistics (Reference §22.13)
# R via PerFit (Tendeiro) or mirt::personfit.
# Run with:  Rscript person_fit_statistics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  PerFit::lz(Y)                        -- Drasgow-Levine-Williams l_z\n")
  cat("  PerFit::HT, PerFit::G                -- Sijtsma H_T + Meijer G\n")
  cat("  mirt::personfit(fit)                 -- l_z + Zh + infit / outfit\n")
}
