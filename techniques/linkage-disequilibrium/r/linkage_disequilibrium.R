# Linkage disequilibrium (Reference Sec 40.26)
# Native R via genetics / snpStats; Python scikit-allel + custom.
# Run with:  Rscript linkage_disequilibrium.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  genetics::LD                    -- D, D', r^2 from genotypes\n")
  cat("  LDheatmap::LDheatmap            -- LD block visualisation\n")
  cat("  snpStats::ld                    -- fast LD across large SNP sets\n")
  cat("  gaston::LD.plot, LD.thin        -- LD-pruning helpers\n")
  cat("Python:\n")
  cat("  scikit-allel::rogers_huff_r, windowed_r_squared\n")
  cat("  hail::hl.ld_matrix              -- distributed LD matrix\n")
  cat("  PLINK --r2 (external)\n")
  cat("Refs: Slatkin (2008) 'Linkage disequilibrium: understanding the evolutionary\n")
  cat("      past and mapping the medical future', Nat Rev Genet; Bulik-Sullivan\n")
  cat("      et al. (2015) LDSC uses r^2 sums (see ld-score-regression).\n")
}
