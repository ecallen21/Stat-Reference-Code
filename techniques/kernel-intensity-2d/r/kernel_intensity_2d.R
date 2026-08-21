# 2D kernel intensity estimation (Reference §23.14)
# R via spatstat.explore::density.ppp or MASS::kde2d.
# Run with:  Rscript kernel_intensity_2d.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spatstat.explore::density.ppp(ppp, sigma, edge=TRUE)   -- intensity (counts / area)\n")
  cat("  spatstat.explore::bw.diggle / bw.CvL / bw.scott          -- bandwidth selectors\n")
  cat("  MASS::kde2d(x, y, h, n)                                  -- density (sums to 1)\n")
  cat("  ks::kde(pts, H=Hns(pts))                                 -- adaptive & plug-in bw\n")
}
