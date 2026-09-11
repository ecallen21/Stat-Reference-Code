"""Datasheets for Datasets (Reference Sec 47.220).

Gebru et al 2018 'Datasheets for Datasets' (revised 2021, CACM).
A structured document accompanying every dataset that answers:

    - MOTIVATION       Why was it collected? Funded by whom?
    - COMPOSITION      Instances, features, splits, targets.
    - COLLECTION       When? How? Any people involved?
    - PREPROCESSING    Cleaning, resampling, filtering.
    - USES             Intended uses; PROHIBITED uses.
    - DISTRIBUTION     License; how to obtain?
    - MAINTENANCE      Who maintains? How to report errors?

Standard practice for research releases + regulated production
pipelines (HIPAA, GDPR). Complements Model Cards (Mitchell 2019).
"""
from __future__ import annotations    # stdlib

import json    # datasheet serialisation


DATASHEET_TEMPLATE = {
    "motivation": {
        "purpose": "",
        "creators": "",
        "funding": "",
    },
    "composition": {
        "n_instances": 0,
        "features": [],
        "target": "",
        "splits": {},
        "sensitive_attributes": [],
        "recommended_use_case": "",
    },
    "collection": {
        "when": "",
        "how": "",
        "consent_process": "",
        "ethical_review": "",
    },
    "preprocessing": {
        "cleaning_steps": [],
        "filtering_criteria": [],
        "known_biases": [],
    },
    "uses": {
        "intended": [],
        "prohibited": [],
        "known_deployments": [],
    },
    "distribution": {
        "license": "",
        "access_url": "",
        "hosting_org": "",
    },
    "maintenance": {
        "maintainer": "",
        "contact": "",
        "update_schedule": "",
        "erratum_process": "",
    },
}


def make_datasheet(dataset_meta):
    """Fill the template with a dataset's metadata; validate required keys."""
    sheet = json.loads(json.dumps(DATASHEET_TEMPLATE))
    for section, fields in dataset_meta.items():
        if section in sheet:
            sheet[section].update(fields)
    return sheet


def check_completeness(sheet):
    """Return list of missing required fields."""
    missing = []
    for section, fields in sheet.items():
        for key, val in fields.items():
            if val == "" or val == [] or val == 0 or val == {}:
                missing.append(f"{section}.{key}")
    return missing


if __name__ == "__main__":
    print("=== Datasheets for Datasets (Gebru et al 2018) ===\n")

    # Well-documented example (mirrors what a proper datasheet contains)
    good_meta = {
        "motivation": {
            "purpose": "Support ML research on clinical trajectory prediction.",
            "creators": "Healthcare AI Consortium",
            "funding": "NIH R01 grant #12345",
        },
        "composition": {
            "n_instances": 50_000,
            "features": ["age", "sex", "labs_last_year", "diagnoses"],
            "target": "30-day readmission (binary)",
            "splits": {"train": 40_000, "val": 5_000, "test": 5_000},
            "sensitive_attributes": ["sex", "race", "zip_code"],
            "recommended_use_case": "Cohort risk stratification",
        },
        "collection": {
            "when": "2018-01 to 2022-12",
            "how": "EHR extract via IRB-approved data-use agreement",
            "consent_process": "IRB waiver of consent (secondary-use)",
            "ethical_review": "IRB protocol #2018-01234",
        },
        "preprocessing": {
            "cleaning_steps": ["dedupe by mrn", "cap outliers at 99.5%"],
            "filtering_criteria": ["age ≥ 18", "at least 1 lab in past year"],
            "known_biases": ["under-represents rural clinics"],
        },
        "uses": {
            "intended": ["research on readmission prediction"],
            "prohibited": ["insurance underwriting", "patient re-identification"],
            "known_deployments": ["ACME Hospital shadow-mode 2024"],
        },
        "distribution": {
            "license": "Restricted Access - Data Use Agreement required",
            "access_url": "https://example.org/dua",
            "hosting_org": "Healthcare AI Consortium",
        },
        "maintenance": {
            "maintainer": "Dr. Data Steward, Healthcare AI Consortium",
            "contact": "steward@example.org",
            "update_schedule": "Annual re-extract",
            "erratum_process": "Submit issue to https://example.org/errata",
        },
    }

    good = make_datasheet(good_meta)
    missing = check_completeness(good)
    print(f"  Well-documented dataset: {len(missing)} missing fields")

    # Under-documented example
    bad_meta = {
        "composition": {"n_instances": 1000, "features": ["x", "y"]},
        "distribution": {"license": "MIT"},
    }
    bad = make_datasheet(bad_meta)
    missing = check_completeness(bad)
    print(f"  Under-documented dataset: {len(missing)} missing fields")
    print(f"  Examples of missing fields: {missing[:5]}")

    print("\n  Real datasheets are 1-4 pages of markdown; the completeness check")
    print("  catches missing motivation, consent, maintainer, and prohibited-use fields.")

    print("\n--- library cross-check (huggingface datasets cards; MLCommons croissant JSON-LD) ---")
