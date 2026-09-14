# Echo State Network (Reference Sec 47.310)
# Native R via reservoir / DeepReser via reticulate; Python via reservoirpy / from-scratch.
# Run with:  Rscript echo_state_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reservoir                     -- ESN with ridge readout in R\n")
  cat("  reticulate + reservoirpy      -- Python-native ESN framework\n")
  cat("Python:\n")
  cat("  reservoirpy                   -- full ESN / DeepESN / conceptor framework\n")
  cat("  easyesn / pyESN               -- classical ESN\n")
  cat("  From-scratch numpy (see echo_state_network.py)\n")
  cat("Refs: Jaeger, H. (2001) 'The echo state approach to analysing and\n")
  cat("      training recurrent neural networks', GMD Report 148;\n")
  cat("      Lukosevicius, M. & Jaeger, H. (2009) 'Reservoir computing\n")
  cat("      approaches to recurrent neural network training',\n")
  cat("      Computer Science Review 3.\n")
}
