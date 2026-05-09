#!/usr/bin/env python3
"""
convert_to_eval_format.py
=========================
Converts our named-slot prediction JSONL files into the numbered-slot
predictions.json format expected by the teammate's eval.py.

Our format (one JSON object per line in .jsonl):
    {"doc_id": "TST3-MUC4-0001", "prediction": {"INCIDENT_TYPE": "BOMBING", ...}}

eval.py format (one dict per document, with numeric slot keys):
    {
      "TST3-MUC4-0001": [
        {"0": "TST3-MUC4-0001", "1": "1", "2": "-", "3": "EL SALVADOR", ...}
      ]
    }

Usage:
    python convert_to_eval_format.py \\
        --pred results/muc/muc4_tst3_qwen_qwen2.5_7b_instruct_zero_shot.jsonl \\
        --out  results/muc/muc4_tst3_qwen_zero_shot_eval_fmt.json

    # Then run teammate's evaluator:
    python eval.py \\
        --pred results/muc/muc4_tst3_qwen_zero_shot_eval_fmt.json \\
        --gold gold_labels.json \\
        --exp-id "#011" --member "Ross" --model "Qwen2.5-7B-Instruct" \\
        --strategy "Independent" --notes "Zero-shot, full 23-slot JSON"
"""

import argparse
import json
from pathlib import Path

# ── Named slot → numeric slot ID mapping (MUC-3/4 official numbering) ─────────
# Slot 0 = MESSAGE ID (doc_id)
# Slot 1 = TEMPLATE ID (template number)
# Slots 2-24 = content slots

NAMED_TO_ID = {
    "INCIDENT_DATE":            2,
    "INCIDENT_LOCATION":        3,
    "INCIDENT_TYPE":            4,
    "INCIDENT_STAGE":           5,
    "INCIDENT_INSTRUMENT_ID":   6,
    "INCIDENT_INSTRUMENT_TYPE": 7,
    "PERP_INCIDENT_CATEGORY":   8,
    "PERP_INDIVIDUAL_ID":       9,
    "PERP_ORGANIZATION_ID":     10,
    "PERP_ORG_CONFIDENCE":      11,
    "PHYS_TGT_ID":              12,
    "PHYS_TGT_TYPE":            13,
    "PHYS_TGT_NUMBER":          14,
    "PHYS_TGT_FOREIGN_NATION":  15,
    "PHYS_TGT_EFFECT":          16,
    "PHYS_TGT_TOTAL_NUMBER":    17,
    "HUM_TGT_NAME":             18,
    "HUM_TGT_DESCRIPTION":      19,
    "HUM_TGT_TYPE":             20,
    "HUM_TGT_NUMBER":           21,
    "HUM_TGT_FOREIGN_NATION":   22,
    "HUM_TGT_EFFECT":           23,
    "HUM_TGT_TOTAL_NUMBER":     24,
}

# eval.py also recognises these alias names — map them just in case
ALIAS_TO_ID = {
    "PERP_ORGANIZATION_CONFIDENCE": 11,
    "PHYS_TGT_EFFECT_OF_INCIDENT":  16,
    "HUM_TGT_EFFECT_OF_INCIDENT":   23,
}
NAMED_TO_ID.update(ALIAS_TO_ID)

NULL_VALUE = "-"


def named_to_numbered(doc_id: str, template_num: int, named: dict) -> dict:
    """
    Convert one named-slot template dict into a numbered-slot dict.

    All 25 slots (0-24) are always present; missing ones default to "-".
    Null-like values (None, "", "null", "none", "n/a") become "-".
    """
    result = {str(i): NULL_VALUE for i in range(25)}
    result["0"] = doc_id
    result["1"] = str(template_num)

    for name, value in named.items():
        slot_id = NAMED_TO_ID.get(name.upper(), NAMED_TO_ID.get(name))
        if slot_id is None:
            continue  # skip unrecognised keys (e.g. "COMMENT")

        # Normalise null-like values to "-"
        if value is None:
            result[str(slot_id)] = NULL_VALUE
        else:
            v = str(value).strip()
            if v.lower() in {"null", "none", "n/a", "na", "", "-"}:
                result[str(slot_id)] = NULL_VALUE
            else:
                result[str(slot_id)] = v

    return result


def convert_jsonl(jsonl_path: str, out_path: str):
    """
    Read our .jsonl predictions and write eval.py-compatible predictions.json.

    Each line in the .jsonl has:
        {"doc_id": "TST3-MUC4-0001", "prediction": <dict or list>}

    If prediction is a dict  → single template (MUC-4 zero-shot style)
    If prediction is a list  → multiple templates (MUC-4 few-shot or MUC-6 style)
    """
    predictions = {}
    n_docs = 0
    n_templates = 0
    n_null_docs = 0

    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            pred   = obj.get("prediction") or obj.get("predicted_template") or {}

            # Normalise to list of templates
            if isinstance(pred, dict):
                templates_raw = [pred]
            elif isinstance(pred, list):
                templates_raw = pred if pred else []
            else:
                templates_raw = []

            # Convert each template
            converted = []
            for i, tmpl in enumerate(templates_raw, start=1):
                if not isinstance(tmpl, dict) or not tmpl:
                    continue
                converted.append(named_to_numbered(doc_id, i, tmpl))

            # If model predicted nothing, produce an empty template
            # (all-null) so the document is represented — it will score all-MIS.
            # Alternatively omit and let eval.py warn. We omit here.
            if not converted:
                n_null_docs += 1

            predictions[doc_id] = converted
            n_docs += 1
            n_templates += len(converted)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, ensure_ascii=False)

    print(f"  Converted {n_docs} documents, {n_templates} templates")
    print(f"  Documents with no valid prediction: {n_null_docs}")
    print(f"  Written to: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert our named-slot JSONL predictions to eval.py numbered-slot format"
    )
    ap.add_argument("--pred", required=True, metavar="PATH",
                    help="Path to our .jsonl predictions file")
    ap.add_argument("--out",  required=True, metavar="PATH",
                    help="Output path for eval.py-compatible predictions.json")
    args = ap.parse_args()

    print(f"\nConverting: {args.pred}")
    convert_jsonl(args.pred, args.out)
    print("\nDone. Now run:")
    print(f"  python eval.py --pred {args.out} --gold gold_labels.json \\")
    print(f"      --exp-id '#011' --member 'Ross' --model 'Qwen2.5-7B-Instruct' \\")
    print(f"      --strategy 'Independent' --notes 'Zero-shot, full 23-slot JSON'")


if __name__ == "__main__":
    main()
