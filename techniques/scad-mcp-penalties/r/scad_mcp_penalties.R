# SCAD + MCP nonconvex penalties (Reference Sec 32.2)
# Native R via ncvreg; Python via reticulate.
# Run with:  Rscript scad_mcp_penalties.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ncvreg                       -- SCAD + MCP for GLMs (Breheny-Huang)\n")
  cat("  glmnet                       -- LASSO / elastic-net (baseline comparison)\n")
  cat("  picasso                       -- fast unified nonconvex penalties\n")
  cat("Python:\n")
  cat("  celer                          -- fast LASSO / SCAD with LLA\n")
  cat("  pyglmnet                       -- GLM elasticnet + limited nonconvex support\n")
  cat("  glmnet-python                  -- Fortran-backed glmnet wrapper\n")
  cat("Refs: Fan, J. & Li, R. (2001) 'Variable selection via nonconcave penalized\n")
  cat("      likelihood and its oracle properties', JASA;\n")
  cat("      Zhang, C.-H. (2010) 'Nearly unbiased variable selection under minimax\n")
  cat("      concave penalty', Annals of Statistics.\n")
}
