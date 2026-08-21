# Knowledge Distillation (Reference §27.x extra)

Train a small **student** to match a large **teacher**'s softmax outputs
(Hinton, Vinyals & Dean 2015). Combined loss:

```
L = α · CE(y_true, p_student)
  + (1 − α) · T² · KL( softmax(z_teacher / T)  ||  softmax(z_student / T) )
```

- **Temperature `T`** softens both distributions, exposing "dark knowledge"
  — the relative probabilities of wrong classes carry more information than
  the argmax alone.
- **`α`** trades hard-label loss for soft-target loss; small `α` (e.g. 0.1–0.3) is common when the teacher is much stronger than the student.
- Scaling by `T²` keeps the gradient magnitude comparable across `T`.

## Variants

| Variant | Distils | Notes |
|---|---|---|
| **Response** (Hinton 2015) | softmax outputs | the original; simplest |
| **Feature** (FitNets, Romero 2014) | intermediate feature maps | helps very deep students |
| **Attention** (Zagoruyko 2016) | attention maps | popular in CNN distillation |
| **Relational** (RKD) | pairwise / triplet relations between representations | preserves structure |
| **Self-distillation** (Zhang 2019, BAN) | teacher and student same size | boosts accuracy over baseline |
| **Data-free** (Chen 2019) | uses synthetic data | works when originals are proprietary |

## When to use

- **Model compression** — deploy a small model with near-teacher quality.
- **Mobile / edge inference** — DistilBERT (~40% smaller BERT, ~97% of task performance).
- **Ensembling into one model** — average many teachers, distil one student.
- **Semi-supervised learning** — teacher labels unlabelled data; student trains on both.
- **LLM alignment shortcuts** — Alpaca / Vicuna: SFT on GPT-4 responses (supervised distillation).

## Files

- `python/knowledge_distillation.py` — from-scratch MLP distillation with temperature-softened KL + hard-label CE mixed by `α`. Demo (3-class 2D task, small student `hidden=(6,)`, 60 labels):
  - Teacher (hidden=(64, 32), all labels): test 0.968.
  - Student, hard labels only: test 0.977.
  - Student, distilled: 0.970.
  - Student, distilled + extra unlabelled transfer set: 0.972.
  On this easy task the small student already saturates; distillation shines when the student is capacity-limited on hard tasks (ImageNet, GLUE) where it typically closes 50–90% of the teacher-student gap.
- `r/knowledge_distillation.R` — `torch::nn_kl_div_loss` + temperature-scaled softmax; Python `torch.nn.functional.kl_div`, `huggingface transformers.DistilBertModel`, `timm --distiller`.

## Assumptions & caveats

- **Temperature choice** — `T = 3–5` typical; too high erases discriminative signal, too low reduces to CE with argmax.
- **α balance** — task-dependent; grid-search on validation.
- **Teacher quality caps student** — a mis-calibrated teacher transfers its bias.
- **Feature dimension mismatch** — feature-distillation needs a projection layer to match student and teacher feature widths.
- **Cascade distillation** — teacher → medium → small often beats one-step distillation.
- **Distillation ≠ compression** — you still need to train the small model from scratch; pruning / quantisation (see `quantization-pruning`) are complementary.

## Related in this repo

- `quantization-pruning` — compression counterpart.
- `transfer-learning` — a related "reuse a pretrained model" recipe.
- `contrastive-learning` — self-supervised pretraining then distil the frozen encoder.
- `deep-mlp-backprop`, `adam-optimizer` — training-loop neighbours.

## Run

```
python techniques/knowledge-distillation/python/knowledge_distillation.py
Rscript techniques/knowledge-distillation/r/knowledge_distillation.R
```

**Refs:** Hinton, G., Vinyals, O. & Dean, J. "Distilling the knowledge in a neural network." *NeurIPS Deep Learning Workshop*, 2015; Romero, A. et al. "FitNets: hints for thin deep nets." *ICLR*, 2015; Sanh, V. et al. "DistilBERT, a distilled version of BERT." *arXiv:1910.01108*, 2019.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
