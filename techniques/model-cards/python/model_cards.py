"""Model cards (Reference Ch 32 MLOps).

Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji &
Gebru (2019) 'Model Cards for Model Reporting.'

Short, structured documentation for a released ML model:
  * MODEL DETAILS: name, version, date, owner.
  * INTENDED USE: primary + out-of-scope uses.
  * FACTORS: relevant demographic / environmental groups.
  * METRICS: accuracy per group, calibration, thresholds.
  * EVALUATION DATA: source, preprocessing, licence.
  * TRAINING DATA: same fields.
  * QUANTITATIVE ANALYSES: unitary + intersectional results.
  * ETHICAL CONSIDERATIONS: risks, mitigations, groups affected.
  * CAVEATS + RECOMMENDATIONS.

Here we implement a MODEL_CARD dataclass, a Markdown formatter, a small
validator that checks required fields are present, and dump a demo
model card for a synthetic churn classifier.
"""
from __future__ import annotations    # stdlib

from dataclasses import dataclass, field, asdict
from typing import Dict, List


REQUIRED_SECTIONS = [
    "model_details", "intended_use", "factors", "metrics",
    "evaluation_data", "training_data", "quantitative_analyses",
    "ethical_considerations", "caveats_and_recommendations",
]


@dataclass
class ModelCard:
    model_details: Dict = field(default_factory=dict)
    intended_use: Dict = field(default_factory=dict)
    factors: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    evaluation_data: Dict = field(default_factory=dict)
    training_data: Dict = field(default_factory=dict)
    quantitative_analyses: Dict = field(default_factory=dict)
    ethical_considerations: Dict = field(default_factory=dict)
    caveats_and_recommendations: Dict = field(default_factory=dict)


def validate(card: ModelCard):
    """Return a list of missing / empty required subsections."""
    missing = []
    for s in REQUIRED_SECTIONS:
        v = getattr(card, s)
        if not v:
            missing.append(s)
    return missing


def to_markdown(card: ModelCard) -> str:
    lines = [f"# Model Card: {card.model_details.get('name', 'unnamed model')} "
              f"v{card.model_details.get('version', 'unknown')}\n"]
    for section in REQUIRED_SECTIONS:
        title = section.replace("_", " ").title()
        lines.append(f"\n## {title}\n")
        v = getattr(card, section)
        if not v:
            lines.append("_TBD_\n")
            continue
        for k, val in v.items():
            if isinstance(val, list):
                lines.append(f"- **{k}**:")
                for item in val:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"- **{k}**: {val}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Model card generator + validator (Mitchell 2019) ===\n")
    card = ModelCard(
        model_details={
            "name": "churn_model",
            "version": "2.0.0",
            "date": "2025-08-31",
            "owner": "risk-modelling@example.org",
            "license": "internal",
            "training_run_id": "run-2025-08-31T12:00",
        },
        intended_use={
            "primary": "Identify subscribers at high risk of churn for outreach.",
            "primary_users": ["Customer-success team"],
            "out_of_scope": [
                "Making credit / employment / housing decisions",
                "Individual-level punitive action",
            ],
        },
        factors={
            "relevant": ["tenure_bucket", "plan_tier"],
            "not_evaluated": ["geographic_region (small samples)"],
        },
        metrics={
            "auc": 0.87,
            "ece": 0.04,
            "threshold_at_10pct_precision": 0.72,
        },
        evaluation_data={"source": "hold-out 20% of Jul-Aug 2025 subscribers",
                          "n": 18_500, "preprocessing": "same as training"},
        training_data={"source": "synthetic proxy for internal Jul 2025 subscribers",
                        "n": 74_000, "preprocessing": "standard-scaler on numeric, target-encode on plan_tier"},
        quantitative_analyses={
            "unitary": "AUC by tenure_bucket: (0-6mo: 0.82, 6-24mo: 0.88, 24+mo: 0.89)",
            "intersectional": "AUC by plan_tier x tenure_bucket documented in appendix",
        },
        ethical_considerations={
            "risks": ["May under-serve short-tenure subscribers if used punitively"],
            "mitigations": ["Guard-rail: score used only for retention outreach, never account termination"],
        },
        caveats_and_recommendations={
            "sensitivity_to_drift": "Retrain quarterly; re-evaluate after any product-launch spike",
            "known_limitations": ["Does not model win-back campaigns",
                                    "Assumes churn label is fully observed within 30 days"],
        },
    )

    missing = validate(card)
    print(f"  Validator: {'OK - all required sections present.' if not missing else f'MISSING: {missing}'}\n")

    md = to_markdown(card)
    print("---- BEGIN MODEL CARD (markdown) ----")
    print(md[:1200] + "\n... (truncated in demo) ...")
    print("---- END MODEL CARD ----")

    # Demo validation: empty card fails.
    print("\n  Empty ModelCard validator: MISSING sections =", validate(ModelCard())[:4], "...")

    print("\n--- library cross-check (google model-cards-toolkit; huggingface Model Cards) ---")
