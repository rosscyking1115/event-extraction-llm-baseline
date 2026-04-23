"""
wikievents_error_analysis.py
Categorises prediction errors from the WikiEvents constrained eval.
Run from ~/team-rg1:
    python scripts/wikievents_error_analysis.py
"""

import json
from collections import defaultdict, Counter

CONSTRAINED_FILE   = "/mnt/parscratch/users/acp25ck/team-rg1/results/wikievents_qwen_eval_constrained_results.jsonl"
UNCONSTRAINED_FILE = "/mnt/parscratch/users/acp25ck/team-rg1/results/wikievents_qwen_eval_results.jsonl"

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_levels(event_type):
    parts = event_type.split('.')
    category = parts[0] if len(parts) > 0 else ''
    subtype  = parts[1] if len(parts) > 1 else ''
    return category, subtype

def categorise(row):
    gold_trigger = row.get('gold_trigger', '').lower().strip()
    gold_type    = row.get('gold_type', '').strip()
    pred_trigger = row.get('pred_trigger', '').lower().strip()
    pred_type    = row.get('pred_type', '').strip()

    t_correct = pred_trigger == gold_trigger
    y_correct = pred_type    == gold_type

    gold_cat, gold_sub = get_levels(gold_type)
    pred_cat, pred_sub = get_levels(pred_type)

    if t_correct and y_correct:
        return 'correct'
    elif t_correct and not y_correct:
        if pred_cat == gold_cat and pred_sub == gold_sub:
            return 'right_trigger_right_cat_sub_wrong_spec'
        elif pred_cat == gold_cat:
            return 'right_trigger_right_category_wrong_subtype'
        else:
            return 'right_trigger_wrong_category'
    elif not t_correct and y_correct:
        return 'wrong_trigger_right_type'
    else:
        # Both wrong — further classify type error
        if pred_cat == gold_cat and pred_sub == gold_sub:
            return 'wrong_trigger_right_cat_sub'
        elif pred_cat == gold_cat:
            return 'wrong_trigger_right_category_wrong_subtype'
        else:
            return 'wrong_trigger_wrong_category'

# ── Analyse constrained results ───────────────────────────────────────────────
rows = load(CONSTRAINED_FILE)
n = len(rows)

category_counts = Counter()
for row in rows:
    cat = categorise(row)
    category_counts[cat] += 1

ORDER = [
    'correct',
    'right_trigger_right_cat_sub_wrong_spec',
    'right_trigger_right_category_wrong_subtype',
    'right_trigger_wrong_category',
    'wrong_trigger_right_type',
    'wrong_trigger_right_cat_sub',
    'wrong_trigger_right_category_wrong_subtype',
    'wrong_trigger_wrong_category',
]

LABELS = {
    'correct':                                   'Correct (trigger + type)',
    'right_trigger_right_cat_sub_wrong_spec':    'Right trigger, right Cat.Sub, wrong Specificity',
    'right_trigger_right_category_wrong_subtype':'Right trigger, right Category, wrong SubType',
    'right_trigger_wrong_category':              'Right trigger, wrong Category entirely',
    'wrong_trigger_right_type':                  'Wrong trigger, right type',
    'wrong_trigger_right_cat_sub':               'Wrong trigger, right Cat.Sub',
    'wrong_trigger_right_category_wrong_subtype':'Wrong trigger, right Category, wrong SubType',
    'wrong_trigger_wrong_category':              'Wrong trigger, wrong Category entirely',
}

print("\n" + "="*70)
print("WIKIEVENTS ERROR ANALYSIS — CONSTRAINED ZERO-SHOT")
print("="*70)
print(f"{'Category':<50} {'Count':>6}  {'%':>6}")
print("-"*70)
for key in ORDER:
    count = category_counts.get(key, 0)
    pct = count / n * 100
    print(f"{LABELS[key]:<50} {count:>6}  {pct:>5.1f}%")
print("-"*70)
print(f"{'Total':<50} {n:>6}  100.0%")

# ── Top confused type pairs ───────────────────────────────────────────────────
print("\n\nTop type confusions (gold → predicted), constrained:")
confusion = Counter()
for row in rows:
    gold = row.get('gold_type', '')
    pred = row.get('pred_type', '')
    if gold != pred:
        confusion[(gold, pred)] += 1

print(f"  {'Gold Type':<42} {'Predicted Type':<42} {'Count':>5}")
print("  " + "-"*90)
for (gold, pred), count in confusion.most_common(15):
    print(f"  {gold:<42} {pred:<42} {count:>5}")

# ── Trigger error analysis ────────────────────────────────────────────────────
print("\n\nTrigger prediction errors (gold → predicted), constrained:")
trigger_errors = Counter()
for row in rows:
    gt = row.get('gold_trigger', '')
    pt = row.get('pred_trigger', '')
    if gt != pt:
        trigger_errors[(gt, pt)] += 1

print(f"  {'Gold Trigger':<25} {'Predicted Trigger':<25} {'Count':>5}")
print("  " + "-"*58)
for (gold, pred), count in trigger_errors.most_common(15):
    print(f"  {gold:<25} {pred:<25} {count:>5}")

# ── Compare unconstrained vs constrained per error category ──────────────────
print("\n\nUnconstrained vs Constrained — error category comparison:")
rows_unc = load(UNCONSTRAINED_FILE)
n_unc = len(rows_unc)

cat_unc = Counter()
for row in rows_unc:
    cat_unc[categorise(row)] += 1

print(f"{'Category':<50} {'Unconstr':>9}  {'Constr':>7}")
print("-"*70)
for key in ORDER:
    u = cat_unc.get(key, 0) / n_unc * 100
    c = category_counts.get(key, 0) / n * 100
    print(f"{LABELS[key]:<50} {u:>8.1f}%  {c:>6.1f}%")

# ── Qualitative examples ──────────────────────────────────────────────────────
print("\n\n--- Qualitative examples (constrained) ---")

example_cats = [
    'correct',
    'right_trigger_right_category_wrong_subtype',
    'right_trigger_wrong_category',
    'wrong_trigger_wrong_category',
]

shown = defaultdict(int)
for row in rows:
    cat = categorise(row)
    if cat in example_cats and shown[cat] < 2:
        print(f"\n[{LABELS[cat]}]")
        print(f"  Sentence:  {row.get('sentence', '')[:180]}")
        print(f"  Gold:      trigger='{row.get('gold_trigger')}' | type={row.get('gold_type')}")
        print(f"  Predicted: trigger='{row.get('pred_trigger')}' | type={row.get('pred_type')}")
        shown[cat] += 1
    if all(shown[c] >= 2 for c in example_cats):
        break

print()
