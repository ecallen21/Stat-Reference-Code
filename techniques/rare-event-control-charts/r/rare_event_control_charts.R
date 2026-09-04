# Rare-event control charts (Reference Sec 37.11)
# Native R via spc; Python custom.
# Run with:  Rscript rare_event_control_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spc                            -- G/T/Bernoulli CUSUM charts (Knoth)\n")
  cat("  qcc                            -- basic p/np/c/u charts\n")
  cat("Python:\n")
  cat("  pyspc + custom                 -- manual\n")
  cat("Refs: Benneyan, J.C. (1998) 'Statistical quality control methods in infection\n")
  cat("      control and hospital epidemiology', Infect Control Hosp Epidemiol;\n")
  cat("      Reynolds, M.R. & Stoumbos, Z.G. (1999) 'A CUSUM chart for monitoring\n")
  cat("      a proportion when inspecting continuously', J Qual Tech.\n")
}
