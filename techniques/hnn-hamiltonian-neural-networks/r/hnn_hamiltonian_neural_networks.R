# Hamiltonian Neural Networks (Reference Sec 47.97)
# Limited R support; Python via torchdyn / hamiltonian-nn.
# Run with:  Rscript hnn_hamiltonian_neural_networks.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class HNN in R; deSolve for ODE integration only\n")
  cat("Python:\n")
  cat("  torchdyn                     -- HNN, LNN, symplectic integrators\n")
  cat("  hamiltonian-nn (Greydanus)   -- author's reference implementation\n")
  cat("  torch + auto-grad            -- straightforward custom HNN\n")
  cat("  from-scratch                 -- see hnn_hamiltonian_neural_networks.py\n")
  cat("Refs: Greydanus, Dzamba & Yosinski (2019) 'Hamiltonian NNs', NeurIPS;\n")
  cat("      Chen, Rubanova, Bettencourt & Duvenaud (2018) 'Neural ODEs', NeurIPS.\n")
}
