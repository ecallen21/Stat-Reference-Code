# Model lineage / provenance (Reference Ch 32 MLOps)
# Native R via igraph for the DAG; Python for the production lineage platforms.
# Run with:  Rscript model_lineage_provenance.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph                      -- DAG storage + ancestor/descendant queries\n")
  cat("  targets                     -- pipeline dependency DAG with caching\n")
  cat("  drake                       -- older pipeline package (superseded by targets)\n")
  cat("Python:\n")
  cat("  openlineage / marquez       -- open-standard lineage events + backend\n")
  cat("  mlflow (model lineage tab)   -- links runs / models / artifacts\n")
  cat("  dbt lineage                  -- data-transform lineage for warehouses\n")
  cat("  datahub, amundsen            -- corporate data-catalogue + lineage platforms\n")
  cat("Refs: Missier, P. et al. (2013) 'The W3C PROV family of specifications';\n")
  cat("      Sculley, D. et al. (2015) 'Hidden Technical Debt in ML Systems'.\n")
}
