# Time-series / tabular / AutoML technique.
# See sibling python/ demo. In R, use reticulate to Python.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("Time-series / tabular / AutoML technique.\n")
cat("No R implementation exists. See the Python demo in this directory.\n")
cat("Reference libraries (Python):\n")
cat("  * darts, neuralforecast, gluonts, pytorch-forecasting\n")
cat("  * pytorch-tabular, autogluon-tabular, tpot, flaml, h2o.automl\n")
