# Network diffusion / contagion models (Reference §24.8)
# R via EpiModel or netdiffuseR.
# Run with:  Rscript network_diffusion.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  EpiModel::netdx / netsim              -- SI / SIS / SIR simulation on dynamic networks\n")
  cat("  netdiffuseR::rdiffnet(t, model='threshold' | 'bernoulli')  -- diffusion of innovations\n")
  cat("  netdiffuseR::exposure(...)            -- exposure-to-adoption analysis\n")
  cat("  igraph::sample_bernoulli(...)         -- edge-percolation baselines\n")
  cat("Python: EoN (Epidemics on Networks); NDlib (multi-model diffusion library)\n")
}
