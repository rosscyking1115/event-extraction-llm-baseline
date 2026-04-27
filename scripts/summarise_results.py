"""
summarise_results.py
Reads all experiment result files and prints a unified comparison table.
Run from ~/team-rg1:
    python scripts/summarise_results.py
"""

import json
import os

RESULTS_DIR = "/mnt/parscratch/users/acp25ck/team-rg1/results"

# Define all known experiments in display order
EXPERIMENTS = [
    {
        "label": "MAVEN unconstrained (Qwen-7B)",
        "file":  "maven_qwen_eval_results.jsonl",
    },
    {
        "label": "MAVEN constrained (Qwen-7B)",
        "file":  "maven_qwen_eval_constrained_results.jsonl",
    },
    {
        "label": "WikiEvents unconstrained (Qwen-7B)",
        "file":  "wikievents_qwen_eval_results.jsonl",
    },
    {
        "label": "WikiEvents constrained (Qwen-7B)",
        "file":  "wikievents_qwen_eval_constrained_results.jsonl",
    },
    {
        "label": "WikiEvents few-shot (Qwen-7B)",
        "file":  "wikievents_qwen_fewshot_results.jsonl",
    },
    {
        "label": "WikiEvents constrained (Llama-3.1-8B)",
        "file":  "wikievents_meta_llama_llama_3.1_8b_instruct_constrained_results.jsonl",
    },
]

def load_results(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_field(row, *candidates):
    """Return the first matching field name found in row (handles schema differences)."""
    for key in candidates:
        if key in row:
            return row[key]
    return False

def compute_metrics(rows):
    n = len(rows)
    if n == 0:
        return None
    valid    = sum(1 for r in rows if get_field(r, 'valid_json'))
    trigger  = sum(1 for r in rows if get_field(r, 'trigger_correct', 'trigger_match'))
    type_acc = sum(1 for r in rows if get_field(r, 'type_correct', 'type_match'))
    both     = sum(1 for r in rows if get_field(r, 'both_correct', 'both_match'))
    return {
        "n":       n,
        "valid":   valid / n,
        "trigger": trigger / n,
        "type":    type_acc / n,
        "both":    both / n,
    }

# ── Print table ───────────────────────────────────────────────────────────────
col_w = 38
print("\n" + "="*97)
print("EXPERIMENT COMPARISON TABLE")
print("="*97)
print(f"{'Experiment':<{col_w}} {'Samples':>8}  {'Valid JSON':>10}  {'Trigger':>8}  {'Type':>8}  {'Both':>8}")
print("-"*97)

for exp in EXPERIMENTS:
    path = os.path.join(RESULTS_DIR, exp["file"])
    if not os.path.exists(path):
        print(f"{exp['label']:<{col_w}} {'—':>8}  {'(not run yet)':>10}")
        continue
    rows = load_results(path)
    m = compute_metrics(rows)
    if m is None:
        print(f"{exp['label']:<{col_w}} {'0':>8}  {'(empty)':>10}")
        continue
    print(
        f"{exp['label']:<{col_w}} {m['n']:>8}  "
        f"{m['valid']:>10.3f}  "
        f"{m['trigger']:>8.3f}  "
        f"{m['type']:>8.3f}  "
        f"{m['both']:>8.3f}"
    )

print("="*97)
print("\nMetrics: exact match accuracy (trigger word, event type, both combined)\n")

# ── Per-type breakdown for WikiEvents ─────────────────────────────────────────
we_constrained = os.path.join(RESULTS_DIR, "wikievents_qwen_eval_constrained_results.jsonl")
if os.path.exists(we_constrained):
    from collections import defaultdict
    rows = load_results(we_constrained)
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        gt = r.get("gold_type", "")
        type_stats[gt]["total"] += 1
        if r.get("type_correct", False):
            type_stats[gt]["correct"] += 1

    print("WikiEvents constrained — per-type accuracy (top 15 by frequency):")
    print(f"  {'Event Type':<45} {'Total':>6}  {'Correct':>8}  {'Acc':>6}")
    print("  " + "-"*70)
    sorted_types = sorted(type_stats.items(), key=lambda x: -x[1]["total"])
    for t, s in sorted_types[:15]:
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {t:<45} {s['total']:>6}  {s['correct']:>8}  {acc:>6.3f}")
    print()
