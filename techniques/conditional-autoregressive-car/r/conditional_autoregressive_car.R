# Conditional Autoregressive (CAR) model (Reference §23.10)
# R via CARBayes, INLA, or spdep::spautolm.
# Run with:  Rscript conditional_autoregressive_car.R

if (sys.nframe() == 0) {
  cat("R packages for CAR / BYM disease mapping:\n")
  cat("  CARBayes::S.CARleroux()          -- Leroux CAR MCMC\n")
  cat("  INLA::inla(y ~ f(id, model='bym', graph=nb))\n")
  cat("  spdep::spautolm(family='CAR')    -- classical CAR MLE\n")
}
