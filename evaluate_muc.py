#!/usr/bin/env python3
"""
MUC Slot-Filling Evaluation Script.

Implements the professor's 4-layer scoring system:
    Layer 1: JSON validity check
    Layer 2: Schema validity check
    Layer 3: Exact match (after normalisation)
    Layer 4: Fuzzy Levenshtein match (threshold 0.8)

Plus TP/FP/FN counting using 5-case rules and micro/macro F1.

Usage:
    python evaluate_muc.py \\
        --predictions results/muc4_tst3_qwen_predictions.jsonl \\
        --gold data/muc4_tst3.json \\
        --dataset muc4 \\
        --model qwen2.5-7b \\
        --prompt_id P1 \\
        --prompt_type zero_shot \\
        --member Ross \\
        --output_csv results/muc4_tst3_qwen_scores.csv

Input formats:
    Gold JSON (from parse_muc34.py or parse_muc6.py):
        List of {doc_id, templates/succession_events, ...}

    Predictions JSONL (one JSON object per line):
        {"doc_id": "...", "prediction": {...}}   # MUC-4: single template dict
        {"doc_id": "...", "prediction": [...]}   # MUC-6: list of event dicts

Output:
    CSV with professor's required columns (27 columns)
    Console summary with micro/macro F1

Scoring rules (5 cases):
    Case 1: Gold null,  Pred null  -> not counted (neither TP nor FP nor FN)
    Case 2: Gold value, Pred value, exact match  -> TP_strict=1, TP_fuzzy=1
    Case 3: Gold value, Pred value, fuzzy match  -> FP_strict=1, FN_strict=1, TP_fuzzy=1
    Case 4: Gold value, Pred value, no match     -> FP_strict=1, FN_strict=1, FP_fuzzy=1, FN_fuzzy=1
    Case 5: Gold value, Pred null                -> FN_strict=1, FN_fuzzy=1
    Case 6: Gold null,  Pred value               -> FP_strict=1, FP_fuzzy=1
"""

import os
import re
import csv
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Slot definitions
# ---------------------------------------------------------------------------

MUC4_EVAL_SLOTS = [
    "INCIDENT_DATE", "INCIDENT_LOCATION", "INCIDENT_TYPE", "INCIDENT_STAGE",
    "INCIDENT_INSTRUMENT_ID", "INCIDENT_INSTRUMENT_TYPE",
    "PERP_INCIDENT_CATEGORY", "PERP_INDIVIDUAL_ID",
    "PERP_ORGANIZATION_ID", "PERP_ORG_CONFIDENCE",
    "PHYS_TGT_ID", "PHYS_TGT_TYPE", "PHYS_TGT_NUMBER",
    "PHYS_TGT_FOREIGN_NATION", "PHYS_TGT_EFFECT", "PHYS_TGT_TOTAL_NUMBER",
    "HUM_TGT_NAME", "HUM_TGT_DESCRIPTION", "HUM_TGT_TYPE",
    "HUM_TGT_NUMBER", "HUM_TGT_FOREIGN_NATION",
    "HUM_TGT_EFFECT", "HUM_TGT_TOTAL_NUMBER",
]

MUC6_EVAL_SLOTS = [
    "succession_org", "post", "vacancy_reason",
    "person_in", "person_out",
    "on_the_job_in", "on_the_job_out",
    "other_org_in", "rel_other_org_in",
]

# Categorical slots where Levenshtein is less meaningful — use exact match only
MUC4_CATEGORICAL = {
    "INCIDENT_TYPE", "INCIDENT_STAGE", "PERP_INCIDENT_CATEGORY",
}
MUC6_CATEGORICAL = {
    "vacancy_reason", "on_the_job_in", "on_the_job_out", "rel_other_org_in",
}

# ---------------------------------------------------------------------------
# Levenshtein / fuzzy matching
# ---------------------------------------------------------------------------

def levenshtein(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    # Use two-row DP
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len2]


def normalised_levenshtein(s1, s2):
    """
    Normalised Levenshtein similarity in [0, 1].
    1.0 = identical, 0.0 = completely different.
    """
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein(s1, s2)
    return 1.0 - dist / max_len


FUZZY_THRESHOLD = 0.8


def fuzzy_match(s1, s2, threshold=FUZZY_THRESHOLD):
    """Return True if normalised Levenshtein similarity >= threshold."""
    return normalised_levenshtein(s1, s2) >= threshold


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------

def normalise_value(val):
    """
    Normalise a slot value for comparison.

    Steps:
        1. Convert to string, strip whitespace
        2. Lowercase
        3. Remove surrounding quotes
        4. Collapse multiple spaces to one
        5. Standardise date formats (crude: normalise separators)
        6. Return None if empty after normalisation
    """
    if val is None:
        return None
    if isinstance(val, list):
        # For list values, normalise each and join
        parts = [normalise_value(v) for v in val]
        parts = [p for p in parts if p]
        return " | ".join(parts) if parts else None

    val = str(val).strip()
    if val in ("-", "*", ""):
        return None

    # Lowercase
    val = val.lower()

    # Strip surrounding quotes
    val = val.strip('"').strip("'")

    # Collapse whitespace
    val = re.sub(r'\s+', ' ', val).strip()

    # Basic date normalisation: standardise separators
    val = re.sub(r'[/\-.]', ' ', val) if re.search(r'\d{1,4}[/\-.]\d{1,2}', val) else val

    # Strip type prefixes for fuzzy matching purposes (e.g. "civilian: " prefix)
    # We keep the full value but also expose the text after ": " for mention matching
    return val if val else None


# ---------------------------------------------------------------------------
# Error type classification
# ---------------------------------------------------------------------------

ERROR_TYPES = [
    "correct",
    "missing_slot",         # Gold has value, pred is null
    "hallucinated_slot",    # Gold is null, pred has value
    "wrong_argument",       # Both filled but no match
    "partial_entity",       # Fuzzy match but not exact
    "date_format",          # Date-related mismatch
    "invalid_json",         # Prediction was not valid JSON
    "schema_error",         # Prediction has wrong keys
    "multiple_values_error",# Pred gave list where single expected or vice versa
    "wrong_event_type",     # Incident type completely wrong
    "over_specific",        # Pred more specific than gold
    "under_specific",       # Pred less specific than gold
    "wrong_template",       # Event extracted for wrong document
    "event_boundary_error", # Event boundaries wrong
]


def classify_error(slot_name, gold_norm, pred_norm, is_exact, is_fuzzy,
                   json_valid, schema_valid):
    """
    Classify the error type for a single slot comparison.
    Returns one of the ERROR_TYPES strings.
    """
    if not json_valid:
        return "invalid_json"
    if not schema_valid:
        return "schema_error"
    if gold_norm is None and pred_norm is None:
        return "correct"  # Both null — not counted but mark correct
    if is_exact:
        return "correct"
    if gold_norm is None and pred_norm is not None:
        return "hallucinated_slot"
    if gold_norm is not None and pred_norm is None:
        return "missing_slot"
    # Both non-null, no exact match
    if is_fuzzy:
        return "partial_entity"
    # Check for date-related slots
    if "DATE" in slot_name.upper():
        return "date_format"
    # Check if types match but mentions differ
    if "TYPE" in slot_name.upper() or "CATEGORY" in slot_name.upper():
        return "wrong_event_type"
    # Pred is longer than gold → over-specific
    if gold_norm and pred_norm:
        if len(pred_norm) > len(gold_norm) * 1.5:
            return "over_specific"
        if len(pred_norm) < len(gold_norm) * 0.5:
            return "under_specific"
    return "wrong_argument"


# ---------------------------------------------------------------------------
# Core scoring for a single slot
# ---------------------------------------------------------------------------

def score_slot(slot_name, gold_val, pred_val,
               json_valid, schema_valid,
               dataset, categorical_slots):
    """
    Score a single slot comparison.

    Returns a dict with all scoring columns.
    """
    gold_norm = normalise_value(gold_val)
    pred_norm = normalise_value(pred_val)

    # Layer 1: JSON valid
    # Layer 2: Schema valid

    # Layer 3: Exact match
    if gold_norm is None and pred_norm is None:
        exact_match = None  # Not applicable
    else:
        exact_match = (gold_norm == pred_norm) and (gold_norm is not None)

    # Layer 4: Fuzzy match
    is_categorical = slot_name in categorical_slots
    if gold_norm is None or pred_norm is None:
        lev_sim = None
        is_fuzzy = False
    elif is_categorical:
        # Categorical: use exact match as fuzzy match too
        lev_sim = 1.0 if gold_norm == pred_norm else 0.0
        is_fuzzy = (gold_norm == pred_norm)
    else:
        lev_sim = normalised_levenshtein(gold_norm, pred_norm)
        is_fuzzy = lev_sim >= FUZZY_THRESHOLD

    # TP/FP/FN using 5-case rules
    if gold_norm is None and pred_norm is None:
        # Case 1: Both null — not counted
        tp_strict = fp_strict = fn_strict = 0
        tp_fuzzy = fp_fuzzy = fn_fuzzy = 0
    elif exact_match:
        # Case 2: Exact match
        tp_strict, fp_strict, fn_strict = 1, 0, 0
        tp_fuzzy, fp_fuzzy, fn_fuzzy = 1, 0, 0
    elif gold_norm is not None and pred_norm is not None and is_fuzzy:
        # Case 3: Fuzzy match only
        tp_strict, fp_strict, fn_strict = 0, 1, 1
        tp_fuzzy, fp_fuzzy, fn_fuzzy = 1, 0, 0
    elif gold_norm is not None and pred_norm is not None:
        # Case 4: Both filled, no match
        tp_strict, fp_strict, fn_strict = 0, 1, 1
        tp_fuzzy, fp_fuzzy, fn_fuzzy = 0, 1, 1
    elif gold_norm is not None and pred_norm is None:
        # Case 5: Missing prediction
        tp_strict, fp_strict, fn_strict = 0, 0, 1
        tp_fuzzy, fp_fuzzy, fn_fuzzy = 0, 0, 1
    else:
        # Case 6: Hallucinated prediction
        tp_strict, fp_strict, fn_strict = 0, 1, 0
        tp_fuzzy, fp_fuzzy, fn_fuzzy = 0, 1, 0

    error_type = classify_error(
        slot_name, gold_norm, pred_norm,
        bool(exact_match), is_fuzzy,
        json_valid, schema_valid
    )

    return {
        "gold_value": str(gold_val) if gold_val is not None else "",
        "predicted_value": str(pred_val) if pred_val is not None else "",
        "normalised_gold": gold_norm or "",
        "normalised_prediction": pred_norm or "",
        "json_valid": int(json_valid),
        "schema_valid": int(schema_valid),
        "exact_match": int(exact_match) if exact_match is not None else 0,
        "levenshtein_similarity": round(lev_sim, 4) if lev_sim is not None else "",
        "fuzzy_match": int(is_fuzzy) if lev_sim is not None else 0,
        "TP_strict": tp_strict,
        "FP_strict": fp_strict,
        "FN_strict": fn_strict,
        "TP_fuzzy": tp_fuzzy,
        "FP_fuzzy": fp_fuzzy,
        "FN_fuzzy": fn_fuzzy,
        "error_type": error_type,
    }


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def compute_f1(tp, fp, fn):
    """Compute precision, recall, F1 from aggregate counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# MUC-4 evaluation
# ---------------------------------------------------------------------------

def evaluate_muc4(gold_data, predictions, args):
    """
    Evaluate MUC-4 predictions.

    gold_data: list of dicts from parse_muc34.py output
    predictions: dict {doc_id: prediction_dict}  (single template per doc)

    Returns list of CSV row dicts.
    """
    rows = []
    gold_map = {d['doc_id']: d for d in gold_data}

    for doc_id, gold_doc in gold_map.items():
        pred_raw = predictions.get(doc_id)

        # JSON validity
        json_valid = pred_raw is not None

        # Schema validity: check all expected keys present
        if json_valid and isinstance(pred_raw, dict):
            expected_keys = set(MUC4_EVAL_SLOTS)
            pred_keys = set(pred_raw.keys())
            schema_valid = expected_keys.issubset(pred_keys)
        else:
            schema_valid = False
            if json_valid and not isinstance(pred_raw, dict):
                # Prediction was valid JSON but wrong type
                pred_raw = {}

        # Get gold templates (use first event template, or null template if none)
        event_templates = [t for t in gold_doc.get('templates', [])
                           if t.get('MESSAGE_TEMPLATE', '*') not in ('*', None)]

        # For now: score against the first gold template
        # (multi-template alignment is a harder problem; we note this as future work)
        if event_templates:
            gold_template = event_templates[0]
        else:
            # Document has no event — gold is all nulls
            gold_template = {slot: None for slot in MUC4_EVAL_SLOTS}

        pred_template = pred_raw if isinstance(pred_raw, dict) else {}

        for slot_name in MUC4_EVAL_SLOTS:
            gold_val = gold_template.get(slot_name)
            pred_val = pred_template.get(slot_name) if json_valid else None

            scores = score_slot(
                slot_name, gold_val, pred_val,
                json_valid, schema_valid,
                "muc4", MUC4_CATEGORICAL
            )

            row = {
                "member": args.member,
                "dataset": "MUC-4",
                "muc_version": "MUC-4",
                "task_type": "terrorism_template",
                "split": gold_doc['split'],
                "doc_id": doc_id,
                "model": args.model,
                "prompt_id": args.prompt_id,
                "prompt_type": args.prompt_type,
                "slot_name": slot_name,
                "notes": "",
            }
            row.update(scores)
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# MUC-6 evaluation
# ---------------------------------------------------------------------------

def evaluate_muc6(gold_data, predictions, args):
    """
    Evaluate MUC-6 ST predictions.

    gold_data: list of dicts from parse_muc6.py output
    predictions: dict {doc_id: list_of_event_dicts}

    For documents with multiple gold events, we align predicted events
    to gold events by minimising total slot-level edit distance (greedy).

    Returns list of CSV row dicts.
    """
    rows = []
    gold_map = {d['doc_id']: d for d in gold_data}

    for doc_id, gold_doc in gold_map.items():
        pred_raw = predictions.get(doc_id)

        json_valid = pred_raw is not None

        # Normalise prediction to list of event dicts
        if not json_valid:
            pred_events = []
            schema_valid = False
        elif isinstance(pred_raw, list):
            pred_events = pred_raw
            # Check schema of first event
            if pred_events and isinstance(pred_events[0], dict):
                schema_valid = all(k in pred_events[0] for k in
                                   ['succession_org', 'post', 'person_in', 'person_out'])
            else:
                schema_valid = len(pred_events) == 0  # Empty list is valid
        elif isinstance(pred_raw, dict):
            # Model returned single event, wrap it
            pred_events = [pred_raw]
            schema_valid = all(k in pred_raw for k in
                               ['succession_org', 'post', 'person_in', 'person_out'])
        else:
            pred_events = []
            schema_valid = False

        gold_events = gold_doc.get('succession_events', [])

        if not gold_events:
            gold_events = [{slot: None for slot in MUC6_EVAL_SLOTS}]

        # Align gold and predicted events
        aligned_pairs = align_events(gold_events, pred_events, MUC6_EVAL_SLOTS)

        for gold_event, pred_event in aligned_pairs:
            for slot_name in MUC6_EVAL_SLOTS:
                gold_val = gold_event.get(slot_name) if gold_event else None
                pred_val = pred_event.get(slot_name) if pred_event else None

                scores = score_slot(
                    slot_name, gold_val, pred_val,
                    json_valid, schema_valid,
                    "muc6", MUC6_CATEGORICAL
                )

                row = {
                    "member": args.member,
                    "dataset": "MUC-6",
                    "muc_version": "MUC-6",
                    "task_type": "succession_template",
                    "split": gold_doc['split'],
                    "doc_id": doc_id,
                    "model": args.model,
                    "prompt_id": args.prompt_id,
                    "prompt_type": args.prompt_type,
                    "slot_name": slot_name,
                    "notes": "",
                }
                row.update(scores)
                rows.append(row)

    return rows


def align_events(gold_events, pred_events, slots):
    """
    Greedily align predicted events to gold events.

    For each gold event, find the best matching predicted event
    (highest number of matching slots). Unmatched gold → (gold, None).
    Unmatched pred → (None, pred).
    """
    if not pred_events:
        return [(g, None) for g in gold_events]
    if not gold_events:
        return [(None, p) for p in pred_events]

    used_pred = [False] * len(pred_events)
    pairs = []

    for gold_ev in gold_events:
        best_idx = -1
        best_score = -1
        for i, pred_ev in enumerate(pred_events):
            if used_pred[i]:
                continue
            if not isinstance(pred_ev, dict):
                continue
            score = sum(
                1 for s in slots
                if normalise_value(gold_ev.get(s)) == normalise_value(pred_ev.get(s))
                and normalise_value(gold_ev.get(s)) is not None
            )
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            used_pred[best_idx] = True
            pairs.append((gold_ev, pred_events[best_idx]))
        else:
            pairs.append((gold_ev, None))

    # Add unmatched predictions
    for i, pred_ev in enumerate(pred_events):
        if not used_pred[i]:
            pairs.append((None, pred_ev))

    return pairs


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(rows, dataset):
    """
    Compute micro and macro F1 from scored rows.
    """
    slots = MUC4_EVAL_SLOTS if dataset == "muc4" else MUC6_EVAL_SLOTS

    # Micro F1: aggregate all counts
    total_tp_s = total_fp_s = total_fn_s = 0
    total_tp_f = total_fp_f = total_fn_f = 0

    # Per-slot counts for macro F1
    slot_tp_s = defaultdict(int)
    slot_fp_s = defaultdict(int)
    slot_fn_s = defaultdict(int)
    slot_tp_f = defaultdict(int)
    slot_fp_f = defaultdict(int)
    slot_fn_f = defaultdict(int)

    error_counts = defaultdict(int)

    for row in rows:
        slot = row['slot_name']
        slot_tp_s[slot] += row['TP_strict']
        slot_fp_s[slot] += row['FP_strict']
        slot_fn_s[slot] += row['FN_strict']
        slot_tp_f[slot] += row['TP_fuzzy']
        slot_fp_f[slot] += row['FP_fuzzy']
        slot_fn_f[slot] += row['FN_fuzzy']

        total_tp_s += row['TP_strict']
        total_fp_s += row['FP_strict']
        total_fn_s += row['FN_strict']
        total_tp_f += row['TP_fuzzy']
        total_fp_f += row['FP_fuzzy']
        total_fn_f += row['FN_fuzzy']

        error_counts[row['error_type']] += 1

    # Micro F1
    micro_p_s, micro_r_s, micro_f1_s = compute_f1(total_tp_s, total_fp_s, total_fn_s)
    micro_p_f, micro_r_f, micro_f1_f = compute_f1(total_tp_f, total_fp_f, total_fn_f)

    # Macro F1 (average per-slot F1)
    slot_f1_s = {}
    slot_f1_f = {}
    for slot in slots:
        _, _, f1_s = compute_f1(slot_tp_s[slot], slot_fp_s[slot], slot_fn_s[slot])
        _, _, f1_f = compute_f1(slot_tp_f[slot], slot_fp_f[slot], slot_fn_f[slot])
        slot_f1_s[slot] = f1_s
        slot_f1_f[slot] = f1_f

    macro_f1_s = sum(slot_f1_s.values()) / len(slots) if slots else 0.0
    macro_f1_f = sum(slot_f1_f.values()) / len(slots) if slots else 0.0

    return {
        "micro_f1_strict": micro_f1_s,
        "micro_p_strict": micro_p_s,
        "micro_r_strict": micro_r_s,
        "micro_f1_fuzzy": micro_f1_f,
        "micro_p_fuzzy": micro_p_f,
        "micro_r_fuzzy": micro_r_f,
        "macro_f1_strict": macro_f1_s,
        "macro_f1_fuzzy": macro_f1_f,
        "slot_f1_strict": slot_f1_s,
        "slot_f1_fuzzy": slot_f1_f,
        "error_counts": dict(error_counts),
        "total_slots_scored": len(rows),
    }


def print_summary(summary, model, dataset):
    """Pretty-print the summary statistics."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — {model} on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"\n  Micro F1 (strict exact):  P={summary['micro_p_strict']:.3f}  "
          f"R={summary['micro_r_strict']:.3f}  F1={summary['micro_f1_strict']:.3f}")
    print(f"  Micro F1 (fuzzy ≥{FUZZY_THRESHOLD}):  P={summary['micro_p_fuzzy']:.3f}  "
          f"R={summary['micro_r_fuzzy']:.3f}  F1={summary['micro_f1_fuzzy']:.3f}")
    print(f"\n  Macro F1 (strict exact):  {summary['macro_f1_strict']:.3f}")
    print(f"  Macro F1 (fuzzy):         {summary['macro_f1_fuzzy']:.3f}")

    print(f"\n  Per-slot F1 (strict | fuzzy):")
    for slot in sorted(summary['slot_f1_strict'].keys()):
        fs = summary['slot_f1_strict'][slot]
        ff = summary['slot_f1_fuzzy'][slot]
        bar = '#' * int(ff * 20)
        print(f"    {slot:<35} {fs:.3f} | {ff:.3f}  {bar}")

    print(f"\n  Error distribution ({summary['total_slots_scored']} slot comparisons):")
    for err, count in sorted(summary['error_counts'].items(),
                              key=lambda x: -x[1]):
        pct = 100 * count / summary['total_slots_scored']
        print(f"    {err:<30} {count:5d} ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_predictions(pred_file):
    """
    Load predictions from a JSONL file.
    Each line: {"doc_id": "...", "prediction": {...} or [...]}

    Returns dict: {doc_id: prediction}
    """
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Line {line_no} is not valid JSON: {e}")
                # Record as invalid
                if 'doc_id' in obj if isinstance(obj, dict) else False:
                    predictions[obj['doc_id']] = None
                continue

            doc_id = obj.get('doc_id') or obj.get('MESSAGE_ID')
            if doc_id is None:
                print(f"  WARNING: Line {line_no} has no doc_id, skipping")
                continue

            pred = obj.get('prediction', obj)
            predictions[doc_id] = pred

    return predictions


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "member", "dataset", "muc_version", "task_type", "split",
    "doc_id", "model", "prompt_id", "prompt_type",
    "slot_name", "gold_value", "predicted_value",
    "normalised_gold", "normalised_prediction",
    "json_valid", "schema_valid",
    "exact_match", "levenshtein_similarity", "fuzzy_match",
    "TP_strict", "FP_strict", "FN_strict",
    "TP_fuzzy", "FP_fuzzy", "FN_fuzzy",
    "error_type", "notes",
]


def write_csv(rows, output_path):
    """Write scoring rows to a CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved -> {output_path}")
    print(f"  {len(rows)} rows written")


# ---------------------------------------------------------------------------
# Empty and majority baselines
# ---------------------------------------------------------------------------

def make_empty_predictions_muc4(gold_data):
    """Return all-null predictions for MUC-4 (empty baseline)."""
    return {d['doc_id']: {slot: None for slot in MUC4_EVAL_SLOTS}
            for d in gold_data}


def make_empty_predictions_muc6(gold_data):
    """Return empty list predictions for MUC-6 (empty baseline)."""
    return {d['doc_id']: [] for d in gold_data}


def make_majority_predictions_muc4(gold_data):
    """
    Return majority-class predictions for MUC-4.
    For categorical slots: use most frequent non-null value.
    For string slots: use null (no majority possible).
    """
    from collections import Counter
    categorical = MUC4_CATEGORICAL
    counts = {slot: Counter() for slot in MUC4_EVAL_SLOTS}

    for doc in gold_data:
        for tmpl in doc.get('templates', []):
            if tmpl.get('MESSAGE_TEMPLATE', '*') in ('*', None):
                continue
            for slot in MUC4_EVAL_SLOTS:
                val = tmpl.get(slot)
                if val is not None:
                    norm = normalise_value(val)
                    if norm:
                        counts[slot][norm] += 1

    majority = {}
    for slot in MUC4_EVAL_SLOTS:
        if slot in categorical and counts[slot]:
            majority[slot] = counts[slot].most_common(1)[0][0]
        else:
            majority[slot] = None

    return {d['doc_id']: majority.copy() for d in gold_data}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MUC slot-filling predictions against gold standard"
    )
    parser.add_argument(
        "--predictions", required=False,
        help="Path to predictions JSONL file. If omitted, runs baseline only."
    )
    parser.add_argument(
        "--gold", required=True,
        help="Path to gold JSON file (from parse_muc34.py or parse_muc6.py)"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["muc4", "muc6"],
        help="Which MUC dataset is being evaluated"
    )
    parser.add_argument(
        "--model", default="unknown",
        help="Model name for CSV metadata"
    )
    parser.add_argument(
        "--prompt_id", default="P1",
        help="Prompt ID for CSV metadata"
    )
    parser.add_argument(
        "--prompt_type", default="zero_shot",
        choices=["zero_shot", "few_shot", "chain_of_thought"],
        help="Prompt type for CSV metadata"
    )
    parser.add_argument(
        "--member", default="Ross",
        help="Team member name for CSV metadata"
    )
    parser.add_argument(
        "--output_csv", default=None,
        help="Path to output CSV file (default: auto-generated in results/)"
    )
    parser.add_argument(
        "--baseline", choices=["empty", "majority", "none"], default="none",
        help="Run a baseline instead of loading predictions"
    )
    args = parser.parse_args()

    # Load gold data
    print(f"Loading gold data from {args.gold}...")
    with open(args.gold, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    print(f"  {len(gold_data)} documents loaded")

    # Load or generate predictions
    if args.baseline == "empty":
        print("Using empty baseline (all nulls)...")
        if args.dataset == "muc4":
            predictions = make_empty_predictions_muc4(gold_data)
        else:
            predictions = make_empty_predictions_muc6(gold_data)
        args.model = "empty_baseline"
        args.prompt_id = "B0"
        args.prompt_type = "baseline"
    elif args.baseline == "majority":
        print("Using majority class baseline...")
        if args.dataset == "muc4":
            predictions = make_majority_predictions_muc4(gold_data)
        else:
            print("  Majority baseline for MUC-6 not yet implemented, using empty")
            predictions = make_empty_predictions_muc6(gold_data)
        args.model = "majority_baseline"
        args.prompt_id = "B1"
        args.prompt_type = "baseline"
    elif args.predictions:
        print(f"Loading predictions from {args.predictions}...")
        predictions = load_predictions(args.predictions)
        print(f"  {len(predictions)} predictions loaded")
    else:
        print("ERROR: Must specify --predictions or --baseline")
        return

    # Evaluate
    print(f"Evaluating {args.dataset.upper()} predictions...")
    if args.dataset == "muc4":
        rows = evaluate_muc4(gold_data, predictions, args)
    else:
        rows = evaluate_muc6(gold_data, predictions, args)

    # Compute summary
    summary = compute_summary(rows, args.dataset)
    print_summary(summary, args.model, args.dataset)

    # Write CSV
    if args.output_csv is None:
        split = gold_data[0]['split'] if gold_data else "test"
        args.output_csv = (f"results/{args.dataset}_{split}_"
                           f"{args.model.replace('/', '-')}_{args.prompt_id}.csv")

    write_csv(rows, args.output_csv)

    # Save summary JSON alongside CSV
    summary_path = args.output_csv.replace('.csv', '_summary.json')
    summary_out = {k: v for k, v in summary.items()
                   if not isinstance(v, dict) or k == 'error_counts'}
    summary_out['slot_f1_strict'] = summary['slot_f1_strict']
    summary_out['slot_f1_fuzzy'] = summary['slot_f1_fuzzy']
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_out, f, indent=2)
    print(f"  Summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
