# Score-Based SDE (Reference Sec 47.212).
#
# Song et al 2021 ICLR. Continuous-time SDE unifying DDPM and score-
# matching; reverse SDE / probability-flow ODE sampling.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Score-Based SDE (Song et al 2021) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * yang-song/score_sde (JAX reference)\n")
cat("  * diffusers.ScoreSdeVeScheduler / ScoreSdeVpScheduler\n")
