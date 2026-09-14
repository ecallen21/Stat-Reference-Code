# Reformer LSH attention (Kitaev-Kaiser-Levskaya 2020 ICLR)
# R: no first-class R package; access via reticulate to
#    `transformers.ReformerModel` or the `trax` reference.
# Python: `transformers.ReformerModel`, `google-research/trax`
#         Reformer, `x_transformers.ReformerAttention`,
#         from-scratch
#
# reticulate::py_install("transformers")
# ref <- reticulate::import("transformers")$ReformerModel
