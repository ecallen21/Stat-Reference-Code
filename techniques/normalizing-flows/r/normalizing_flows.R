# Normalising flows (Reference §27.x extra)
# R via torch + custom bijections, or reticulate + Python.
# Run with:  Rscript normalizing_flows.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: manual coupling / autoregressive layers with change-of-variables loss\n")
  cat("Python:\n")
  cat("  nflows -- comprehensive flow library (coupling, autoregressive, spline flows)\n")
  cat("  FrEIA -- flexible framework for invertible neural nets\n")
  cat("  tensorflow-probability bijectors (RealNVP, MAF, IAF, Neural-Spline)\n")
  cat("  pyro.distributions.transforms\n")
  cat("Flow families:\n")
  cat("  * RealNVP / Glow -- coupling with alternating masks; fast forward + inverse\n")
  cat("  * MAF (Papamakarios 2017) -- autoregressive; fast density; slow sampling\n")
  cat("  * IAF (Kingma 2016) -- inverse autoregressive; fast sampling; slow density\n")
  cat("  * Neural Spline (Durkan 2019) -- expressive rational-quadratic splines\n")
  cat("  * Continuous / Neural ODE flows (Chen 2018)\n")
  cat("Alternatives: diffusion (see diffusion-model), VAEs (see variational-autoencoder), GANs.\n")
}
