# Population-Based Training (Reference Sec 47.277)
# Native R via reticulate + Ray Tune; Python via Ray Tune / from-scratch.
# Run with:  Rscript population_based_training.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate::use_python + Ray Tune (PBT is Python-native)\n")
  cat("  torch (R) + custom loop for training runs\n")
  cat("Python:\n")
  cat("  Ray Tune PopulationBasedTraining\n")
  cat("  Ax Service API (partial PBT semantics)\n")
  cat("  From-scratch (see population_based_training.py)\n")
  cat("Refs: Jaderberg, M., Dalibard, V., Osindero, S., et al (2017)\n")
  cat("      'Population Based Training of Neural Networks', arXiv:1711.09846.\n")
}
