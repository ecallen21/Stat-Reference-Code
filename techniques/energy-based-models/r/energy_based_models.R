# Energy-based models (Reference §27.x extra)
# R via reticulate + Python.
# Run with:  Rscript energy_based_models.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  no dedicated library; roll your own in torch or JAX.\n")
  cat("  score-based generative modelling (Song 2019/2021) shares infrastructure:\n")
  cat("    * score-based-sde -- score matching + Langevin sampling\n")
  cat("    * SDE-diffusers -- diffusion as EBM special case\n")
  cat("Reference implementations:\n")
  cat("  * Restricted Boltzmann Machines (Hinton 2002; CD-k origin)\n")
  cat("  * JEM (Grathwohl 2020) -- joint energy-based classifier\n")
  cat("  * Score matching (Hyvarinen 2005) -- avoids partition function\n")
  cat("  * Diffusion (see diffusion-model) -- SOTA generative EBM subfamily\n")
}
