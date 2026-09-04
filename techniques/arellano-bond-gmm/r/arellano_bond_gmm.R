# Arellano-Bond difference GMM (Reference Sec 35.3)
# Native R via plm::pgmm; Python via linearmodels.
# Run with:  Rscript arellano_bond_gmm.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  plm::pgmm                    -- Arellano-Bond + Blundell-Bond system GMM\n")
  cat("  panelvar                      -- panel VAR with GMM\n")
  cat("Python:\n")
  cat("  linearmodels.panel.PanelGMM  -- Kevin Sheppard's implementation\n")
  cat("  pydynpd                        -- dynamic panel GMM (community)\n")
  cat("Refs: Arellano, M. & Bond, S. (1991) 'Some tests of specification for panel\n")
  cat("      data: Monte Carlo evidence and an application to employment equations',\n")
  cat("      Review of Economic Studies;\n")
  cat("      Blundell, R. & Bond, S. (1998) 'Initial conditions and moment restrictions\n")
  cat("      in dynamic panel data models', J. Econometrics.\n")
}
