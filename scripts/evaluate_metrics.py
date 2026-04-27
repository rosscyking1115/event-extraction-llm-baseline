"""
evaluate_metrics.py
Computes comprehensive evaluation metrics for all event extraction experiments.
Includes:
  - Exact match (precision, recall, F1)
  - Fuzzy match using Levenshtein distance (normalised)
  - Hierarchical partial credit for WikiEvents type matching
  - Micro F1 and Macro F1 per event type
  - Slot-level evaluation template

Run from ~/team-rg1:
    python scripts/evaluate_metrics.py
"""

import json
import os
from collections import defaultdict, Counter
import math

RESULTS_DIR  = "/mnt/parscratch/users/acp25ck/team-rg1/results"
TEMPLATE_DIR = "/mnt/parscratch/users/acp25ck/team-rg1/results/templates"

EXPERIMENTS = [
    {
        "label": "MAVEN unconstrained (Qwen-7B)",
        "file":  "maven_qwen_eval_results.jsonl",
        "dataset": "maven",
    },
    {
        "label": "MAVEN constrained (Qwen-7B)",
        "file":  "maven_qwen_eval_constrained_results.jsonl",
        "dataset": "maven",
    },
    {
        "label": "WikiEvents unconstrained (Qwen-7B)",
        "file":  "wikievents_qwen_eval_results.jsonl",
        "dataset": "wikievents",
    },
    {
        "label": "WikiEvents constrained (Qwen-7B)",
        "file":  "wikievents_qwen_eval_constrained_results.jsonl",
        "dataset": "wikievents",
    },
    {
        "label": "WikiEvents few-shot (Qwen-7B)",
        "file":  "wikievents_qwen_fewshot_results.jsonl",
        "dataset": "wikievents",
    },
    {
        "label": "WikiEvents constrained (Llama-3.1-8B)",
        "file":  "wikievents_meta_llama_llama_3.1_8b_instruct_constrained_results.jsonl",
        "dataset": "wikievents",
    },
]

# ── Levenshtein distance ──────────────────────────────────────────────────────
def levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,       # deletion
                matrix[i][j-1] + 1,       # insertion
                matrix[i-1][j-1] + cost   # substitution
            )
    return matrix[len(s1)][len(s2)]

def normalised_levenshtein(s1: str, s2: str) -> float:
    """Normalised similarity: 1.0 = identical, 0.0 = completely different."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein(s1, s2)
    return 1.0 - dist / max_len

def fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Return True if normalised similarity >= threshold."""
    return normalised_levenshtein(s1, s2) >= threshold

# ── Hierarchical type partial credit ─────────────────────────────────────────
def type_partial_credit(gold: str, pred: str) -> float:
    """
    For WikiEvents 3-level types (Cat.Sub.Spec):
      Exact match        → 1.0
      Right Cat + Sub    → 0.67
      Right Cat only     → 0.33
      No match           → 0.0
    For flat types (MAVEN): exact = 1.0, else 0.0
    """
    if gold == pred:
        return 1.0
    gold_parts = gold.split('.')
    pred_parts = pred.split('.')
    if len(gold_parts) < 2:
        return 0.0
    if gold_parts[0] == pred_parts[0] and len(pred_parts) > 1 and gold_parts[1] == pred_parts[1]:
        return 0.67
    if gold_parts[0] == pred_parts[0]:
        return 0.33
    return 0.0

# ── F1 helpers ────────────────────────────────────────────────────────────────
def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def get_field(row, *candidates):
    for key in candidates:
        if key in row:
            return row[key]
    return False

# ── Load results ──────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate(rows, dataset='wikievents'):
    n = len(rows)
    if n == 0:
        return None

    # Per-row slot records
    slot_records = []

    # Aggregates for micro F1
    trigger_tp = trigger_fp = trigger_fn = 0
    type_tp    = type_fp    = type_fn    = 0
    both_tp    = both_fp    = both_fn    = 0

    # Per-type counts for macro F1
    type_class_tp = defaultdict(int)
    type_class_fp = defaultdict(int)
    type_class_fn = defaultdict(int)

    # Fuzzy match counts
    trigger_fuzzy_match = 0
    type_fuzzy_match    = 0

    # Partial credit sums
    trigger_lev_sim_total = 0.0
    type_partial_total    = 0.0

    for row in rows:
        gold_trigger = row.get('gold_trigger', '').lower().strip()
        gold_type    = row.get('gold_type', '').strip()
        pred_trigger = row.get('pred_trigger', '').lower().strip()
        pred_type    = row.get('pred_type', '').strip()

        # Exact match
        t_exact = pred_trigger == gold_trigger
        y_exact = pred_type    == gold_type
        b_exact = t_exact and y_exact

        # Levenshtein similarity
        t_lev = normalised_levenshtein(gold_trigger, pred_trigger)
        y_lev = normalised_levenshtein(gold_type,    pred_type)

        # Fuzzy match (threshold 0.8)
        t_fuzzy = fuzzy_match(gold_trigger, pred_trigger, threshold=0.8)
        y_fuzzy = fuzzy_match(gold_type,    pred_type,    threshold=0.8)

        # Partial credit for type
        y_partial = type_partial_credit(gold_type, pred_type)

        # Trigger micro F1
        if t_exact:
            trigger_tp += 1
        else:
            trigger_fp += 1
            trigger_fn += 1

        # Type micro F1
        if y_exact:
            type_tp += 1
            type_class_tp[gold_type] += 1
        else:
            type_fp += 1
            type_fn += 1
            type_class_fp[pred_type] += 1
            type_class_fn[gold_type] += 1

        # Both micro F1
        if b_exact:
            both_tp += 1
        else:
            both_fp += 1
            both_fn += 1

        if t_fuzzy:
            trigger_fuzzy_match += 1
        if y_fuzzy:
            type_fuzzy_match += 1

        trigger_lev_sim_total += t_lev
        type_partial_total    += y_partial

        slot_records.append({
            'gold_trigger':    gold_trigger,
            'pred_trigger':    pred_trigger,
            'trigger_exact':   t_exact,
            'trigger_lev_sim': round(t_lev, 3),
            'trigger_fuzzy':   t_fuzzy,
            'gold_type':       gold_type,
            'pred_type':       pred_type,
            'type_exact':      y_exact,
            'type_lev_sim':    round(y_lev, 3),
            'type_fuzzy':      y_fuzzy,
            'type_partial':    round(y_partial, 2),
            'both_exact':      b_exact,
        })

    # Micro F1
    t_prec, t_rec, t_f1 = compute_f1(trigger_tp, trigger_fp, trigger_fn)
    y_prec, y_rec, y_f1 = compute_f1(type_tp,    type_fp,    type_fn)
    b_prec, b_rec, b_f1 = compute_f1(both_tp,    both_fp,    both_fn)

    # Macro F1 for type
    all_types = set(list(type_class_tp.keys()) + list(type_class_fn.keys()))
    macro_f1_scores = []
    for t in all_types:
        tp = type_class_tp[t]
        fp = type_class_fp[t]
        fn = type_class_fn[t]
        _, _, f1 = compute_f1(tp, fp, fn)
        macro_f1_scores.append(f1)
    macro_type_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0

    return {
        'n': n,
        # Exact match
        'trigger_exact_acc':   trigger_tp / n,
        'type_exact_acc':      type_tp / n,
        'both_exact_acc':      both_tp / n,
        # Fuzzy match
        'trigger_fuzzy_acc':   trigger_fuzzy_match / n,
        'type_fuzzy_acc':      type_fuzzy_match / n,
        # Average Levenshtein similarity
        'trigger_avg_lev':     trigger_lev_sim_total / n,
        'type_avg_lev':        type_partial_total / n,  # partial credit for type
        # Micro F1
        'trigger_precision':   t_prec,
        'trigger_recall':      t_rec,
        'trigger_micro_f1':    t_f1,
        'type_precision':      y_prec,
        'type_recall':         y_rec,
        'type_micro_f1':       y_f1,
        'both_micro_f1':       b_f1,
        # Macro F1
        'type_macro_f1':       macro_type_f1,
        # Slot records
        'slot_records':        slot_records,
    }

# ── Print results ─────────────────────────────────────────────────────────────
os.makedirs(TEMPLATE_DIR, exist_ok=True)

print("\n" + "="*110)
print("COMPREHENSIVE EVALUATION — EXACT MATCH, FUZZY MATCH, F1")
print("="*110)

# Table 1: Exact match + Fuzzy match
print(f"\n{'Experiment':<42} {'N':>5}  {'Trig Exact':>10}  {'Trig Fuzzy':>10}  {'Type Exact':>10}  {'Type Fuzzy':>10}  {'Both':>6}")
print("-"*100)

all_metrics = {}
for exp in EXPERIMENTS:
    path = os.path.join(RESULTS_DIR, exp['file'])
    if not os.path.exists(path):
        print(f"{exp['label']:<42} {'—':>5}  {'(not run yet)':>10}")
        continue
    rows = load(path)
    m = evaluate(rows, exp['dataset'])
    if m is None:
        continue
    all_metrics[exp['label']] = m
    print(
        f"{exp['label']:<42} {m['n']:>5}  "
        f"{m['trigger_exact_acc']:>10.3f}  "
        f"{m['trigger_fuzzy_acc']:>10.3f}  "
        f"{m['type_exact_acc']:>10.3f}  "
        f"{m['type_fuzzy_acc']:>10.3f}  "
        f"{m['both_exact_acc']:>6.3f}"
    )

# Table 2: F1 scores
print(f"\n\n{'Experiment':<42} {'N':>5}  {'Trig P':>7}  {'Trig R':>7}  {'Trig F1':>8}  {'Type F1':>8}  {'Macro F1':>9}  {'Both F1':>8}")
print("-"*105)
for exp in EXPERIMENTS:
    if exp['label'] not in all_metrics:
        continue
    m = all_metrics[exp['label']]
    print(
        f"{exp['label']:<42} {m['n']:>5}  "
        f"{m['trigger_precision']:>7.3f}  "
        f"{m['trigger_recall']:>7.3f}  "
        f"{m['trigger_micro_f1']:>8.3f}  "
        f"{m['type_micro_f1']:>8.3f}  "
        f"{m['type_macro_f1']:>9.3f}  "
        f"{m['both_micro_f1']:>8.3f}"
    )

# Table 3: Levenshtein similarity + type partial credit
print(f"\n\n{'Experiment':<42} {'N':>5}  {'Trig Lev Sim':>13}  {'Type Partial Credit':>20}")
print("-"*85)
for exp in EXPERIMENTS:
    if exp['label'] not in all_metrics:
        continue
    m = all_metrics[exp['label']]
    print(
        f"{exp['label']:<42} {m['n']:>5}  "
        f"{m['trigger_avg_lev']:>13.3f}  "
        f"{m['type_avg_lev']:>20.3f}"
    )

print("\n")

# ── Slot-level template output ────────────────────────────────────────────────
print("="*80)
print("SLOT-LEVEL EVALUATION TEMPLATE (first 10 rows per experiment)")
print("="*80)

for exp in EXPERIMENTS:
    if exp['label'] not in all_metrics:
        continue
    m = all_metrics[exp['label']]
    print(f"\n--- {exp['label']} ---")
    print(f"{'#':<5} {'Gold Trigger':<20} {'Pred Trigger':<20} {'Lev':>5} {'FM':>3} | {'Gold Type':<35} {'Pred Type':<35} {'PC':>5} {'FM':>3} {'EX':>3}")
    print("-"*140)
    for i, r in enumerate(m['slot_records'][:10]):
        print(
            f"{i+1:<5} "
            f"{r['gold_trigger']:<20} {r['pred_trigger']:<20} "
            f"{r['trigger_lev_sim']:>5.2f} {'Y' if r['trigger_fuzzy'] else 'N':>3} | "
            f"{r['gold_type']:<35} {r['pred_type']:<35} "
            f"{r['type_partial']:>5.2f} {'Y' if r['type_fuzzy'] else 'N':>3} {'Y' if r['type_exact'] else 'N':>3}"
        )

    # Save full template to file
    template_path = os.path.join(TEMPLATE_DIR, exp['file'].replace('.jsonl', '_template.tsv'))
    with open(template_path, 'w') as f:
        f.write("idx\tgold_trigger\tpred_trigger\ttrigger_lev_sim\ttrigger_fuzzy\ttrigger_exact\tgold_type\tpred_type\ttype_partial_credit\ttype_fuzzy\ttype_exact\tboth_exact\n")
        for i, r in enumerate(m['slot_records']):
            f.write(
                f"{i+1}\t{r['gold_trigger']}\t{r['pred_trigger']}\t"
                f"{r['trigger_lev_sim']}\t{r['trigger_fuzzy']}\t{r['trigger_exact']}\t"
                f"{r['gold_type']}\t{r['pred_type']}\t"
                f"{r['type_partial']}\t{r['type_fuzzy']}\t{r['type_exact']}\t{r['both_exact']}\n"
            )
    print(f"  [Full template saved: {template_path}]")

print("\nDone.")
