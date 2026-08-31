# Neural ODEs (Reference §27.x extra)
# R via deSolve for classical ODEs, or reticulate + Python.
# Run with:  Rscript neural_ode.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  deSolve::ode(y=z0, times=..., func=..., parms=...)  -- classical ODE solvers\n")
  cat("  torch::nn_module (manual Euler / RK4 forward)\n")
  cat("  reticulate + torchdiffeq (Chen et al. 2018) -- PyTorch NODE library\n")
  cat("Python:\n")
  cat("  torchdiffeq.odeint / odeint_adjoint     -- PyTorch NODEs with adjoint memory\n")
  cat("  diffrax                                  -- JAX equivalent; SOTA solvers, SDEs, CDEs\n")
  cat("  jaxdemo neural-ode notebooks\n")
  cat("Applications:\n")
  cat("  * Continuous normalising flows (Chen 2018; Grathwohl 2018 FFJORD)\n")
  cat("  * Time series with irregular sampling (Rubanova 2019 ODE-RNN)\n")
  cat("  * Neural SDEs / CDEs (Kidger, Morrill 2020) for path-dependent models\n")
  cat("  * Physics-informed neural networks (Raissi 2019)\n")
}
