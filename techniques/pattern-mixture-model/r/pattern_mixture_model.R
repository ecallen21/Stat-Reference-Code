# Pattern-mixture models for MNAR data (Reference §16.x extra)
# R via mice + delta-adjustment, or jomo, or SensMice.
# Run with:  Rscript pattern_mixture_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mice::mice(dat, method='norm')  +  post = 'imp[[k]][,j] <- imp[[k]][,j] + delta'\n")
  cat("  jomo::jomo1ranmix(...)          -- joint MI for MNAR mixtures\n")
  cat("  SensMice::sens.mice(...)        -- delta-tipping-point wrapper\n")
  cat("  RefBasedMI::refBasedMI(...)     -- reference-based MI (Carpenter-Kenward)\n")
  cat("  gsheet::pattern.mixture / SAS PROC MI CS option (control-based imputation)\n")
}
