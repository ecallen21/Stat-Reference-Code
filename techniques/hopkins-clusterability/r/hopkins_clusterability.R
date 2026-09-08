# Hopkins clusterability statistic (Reference Sec 47.73)
# Native R via clustertend / factoextra; Python via pyclustertend / hopkins.
# Run with:  Rscript hopkins_clusterability.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  clustertend::hopkins          -- Hopkins with kd-tree neighbours\n")
  cat("  factoextra::get_clust_tendency -- Hopkins + VAT (ordered dissim image)\n")
  cat("  cluster::silhouette           -- companion cluster-validation index\n")
  cat("Python:\n")
  cat("  pyclustertend.hopkins         -- direct port of R clustertend\n")
  cat("  hopkins                       -- lightweight PyPI package\n")
  cat("  from-scratch                  -- see hopkins_clusterability.py\n")
  cat("Refs: Hopkins & Skellam (1954) Annals of Botany 18(2); Lawson & Jurs (1990)\n")
  cat("      J Chem Inf Comput Sci 30(1).\n")
}
