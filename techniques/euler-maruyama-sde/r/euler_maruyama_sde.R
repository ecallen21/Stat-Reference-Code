# Euler-Maruyama SDE simulation (Reference Sec 47.57)
# Native R via Sim.DiffProc / yuima / sde; Python via sdeint / diffrax.
# Run with:  Rscript euler_maruyama_sde.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Sim.DiffProc::snssde1d / snssde2d  -- Euler / Milstein / SRK schemes\n")
  cat("  yuima::simulate                    -- Levy + jump-diffusion extension\n")
  cat("  sde::sde.sim                       -- Iacus 2008 textbook helpers\n")
  cat("Python:\n")
  cat("  sdeint                             -- Ito & Stratonovich integrators\n")
  cat("  diffrax                            -- JAX-native SDE / ODE library\n")
  cat("  torchsde                           -- backprop through SDE solutions\n")
  cat("  from-scratch                       -- see euler_maruyama_sde.py\n")
  cat("Refs: Maruyama (1955) Rend Circ Mat Palermo 4; Kloeden & Platen (1992)\n")
  cat("      'Numerical Solution of Stochastic Differential Equations', Springer.\n")
}
