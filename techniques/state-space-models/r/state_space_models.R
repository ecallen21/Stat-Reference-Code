# State-space models: S4 / Mamba (Reference §27.x extra)
# R via KFAS + custom scan, or reticulate + Python.
# Run with:  Rscript state_space_models.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  KFAS::SSModel + KFS(...)               -- classical linear-Gaussian state space + Kalman\n")
  cat("  dlm, MARSS                              -- dynamic linear models\n")
  cat("  reticulate + mamba-ssm / s4d           -- modern selective SSMs\n")
  cat("Modern architectures:\n")
  cat("  * S4 (Gu 2022)           -- HiPPO init + diagonal parametrisation + FFT convolution\n")
  cat("  * S5 (Smith 2023)        -- simplified diagonal S4\n")
  cat("  * Mamba (Gu-Dao 2023)   -- selective (input-dependent) SSM; O(T) attention alternative\n")
  cat("  * Mamba-2 (Dao-Gu 2024) -- structured state-space duality; connects SSMs and attention\n")
  cat("  * Griffin / RecurrentGemma (Botev 2024) -- hybrid SSM + attention\n")
  cat("  * RWKV (Peng 2023)       -- linear attention alternative\n")
  cat("Applications: long-context language modelling, DNA/protein, audio, time-series forecasting.\n")
}
