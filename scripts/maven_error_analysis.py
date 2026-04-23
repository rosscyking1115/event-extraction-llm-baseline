"""
maven_error_analysis.py
Categorises prediction errors from the MAVEN unconstrained and constrained evals.
Run from ~/team-rg1:
    python scripts/maven_error_analysis.py
"""

import json
from collections import defaultdict, Counter

UNCONSTRAINED_FILE = "/mnt/parscratch/users/acp25ck/team-rg1/results/maven_qwen_eval_results.jsonl"
CONSTRAINED_FILE   = "/mnt/parscratch/users/acp25ck/team-rg1/results/maven_qwen_eval_constrained_results.jsonl"

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_fields(row):
    """Handle both old (trigger_match) and new (trigger_correct) field names."""
    gold_trigger = row.get('gold_trigger', '').lower().strip()
    gold_type    = row.get('gold_type', '').strip()
    pred_trigger = row.get('pred_trigger', '').lower().strip()
    pred_type    = row.get('pred_type', '').strip()
    t_correct = row.get('trigger_correct', row.get('trigger_match', False))
    y_correct = row.get('type_correct',    row.get('type_match',    False))
    return gold_trigger, gold_type, pred_trigger, pred_type, t_correct, y_correct

def categorise(row):
    gold_trigger, gold_type, pred_trigger, pred_type, t_correct, y_correct = get_fields(row)

    if t_correct and y_correct:
        return 'correct'
    elif t_correct and not y_correct:
        return 'right_trigger_wrong_type'
    elif not t_correct and y_correct:
        return 'wrong_trigger_right_type'
    else:
        return 'wrong_trigger_wrong_type'

def analyse(rows, label):
    n = len(rows)
    cat_counts = Counter(categorise(r) for r in rows)

    ORDER = [
        'correct',
        'right_trigger_wrong_type',
        'wrong_trigger_right_type',
        'wrong_trigger_wrong_type',
    ]
    LABELS = {
        'correct':                 'Correct (trigger + type)',
        'right_trigger_wrong_type':'Right trigger, wrong type',
        'wrong_trigger_right_type':'Wrong trigger, right type',
        'wrong_trigger_wrong_type':'Wrong trigger, wrong type',
    }

    print(f"\n{'='*65}")
    print(f"MAVEN ERROR ANALYSIS — {label}")
    print(f"{'='*65}")
    print(f"{'Category':<40} {'Count':>6}  {'%':>6}")
    print("-"*55)
    for key in ORDER:
        count = cat_counts.get(key, 0)
        print(f"{LABELS[key]:<40} {count:>6}  {count/n*100:>5.1f}%")
    print("-"*55)
    print(f"{'Total':<40} {n:>6}  100.0%")

    # Type confusions
    print(f"\nTop type confusions (gold → predicted):")
    confusion = Counter()
    for row in rows:
        _, gold_type, _, pred_type, _, y_correct = get_fields(row)
        if not y_correct:
            confusion[(gold_type, pred_type)] += 1
    print(f"  {'Gold Type':<30} {'Predicted Type':<30} {'Count':>5}")
    print("  " + "-"*68)
    for (gold, pred), count in confusion.most_common(10):
        print(f"  {gold:<30} {pred:<30} {count:>5}")

    # Trigger errors
    print(f"\nTop trigger errors (gold → predicted):")
    trigger_errors = Counter()
    for row in rows:
        gold_trigger, _, pred_trigger, _, t_correct, _ = get_fields(row)
        if not t_correct:
            trigger_errors[(gold_trigger, pred_trigger)] += 1
    print(f"  {'Gold Trigger':<25} {'Predicted Trigger':<25} {'Count':>5}")
    print("  " + "-"*58)
    for (gold, pred), count in trigger_errors.most_common(10):
        print(f"  {gold:<25} {pred:<25} {count:>5}")

    # Qualitative examples
    print(f"\n--- Qualitative examples ---")
    shown = defaultdict(int)
    cats_to_show = ['correct', 'right_trigger_wrong_type', 'wrong_trigger_wrong_type']
    for row in rows:
        cat = categorise(row)
        if cat in cats_to_show and shown[cat] < 2:
            gold_trigger, gold_type, pred_trigger, pred_type, _, _ = get_fields(row)
            print(f"\n[{LABELS[cat]}]")
            sentence = row.get('sentence', row.get('text', ''))[:180]
            print(f"  Sentence:  {sentence}")
            print(f"  Gold:      trigger='{gold_trigger}' | type={gold_type}")
            print(f"  Predicted: trigger='{pred_trigger}' | type={pred_type}")
            shown[cat] += 1
        if all(shown[c] >= 2 for c in cats_to_show):
            break

# ── Run analysis on both files ────────────────────────────────────────────────
rows_unc = load(UNCONSTRAINED_FILE)
rows_con = load(CONSTRAINED_FILE)

analyse(rows_unc, "UNCONSTRAINED ZERO-SHOT")
analyse(rows_con, "CONSTRAINED ZERO-SHOT")

# ── Cross-condition comparison ────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print("MAVEN — UNCONSTRAINED vs CONSTRAINED COMPARISON")
print(f"{'='*65}")

ORDER = ['correct', 'right_trigger_wrong_type', 'wrong_trigger_right_type', 'wrong_trigger_wrong_type']
LABELS = {
    'correct':                 'Correct (trigger + type)',
    'right_trigger_wrong_type':'Right trigger, wrong type',
    'wrong_trigger_right_type':'Wrong trigger, right type',
    'wrong_trigger_wrong_type':'Wrong trigger, wrong type',
}

cat_unc = Counter(categorise(r) for r in rows_unc)
cat_con = Counter(categorise(r) for r in rows_con)
n_unc, n_con = len(rows_unc), len(rows_con)

print(f"{'Category':<40} {'Unconstr':>9}  {'Constr':>7}")
print("-"*60)
for key in ORDER:
    u = cat_unc.get(key, 0) / n_unc * 100
    c = cat_con.get(key, 0) / n_con * 100
    print(f"{LABELS[key]:<40} {u:>8.1f}%  {c:>6.1f}%")

# ── Cross-dataset comparison summary ──────────────────────────────────────────
print(f"\n\n{'='*65}")
print("MAVEN vs WIKIEVENTS — ERROR PATTERN COMPARISON")
print(f"{'='*65}")
print("""
Key structural differences:
  MAVEN:      flat label space (~168 types), sentence-level, detection only
  WikiEvents: 3-level hierarchy (49 types), document-level, with arguments

Expected finding:
  - MAVEN unconstrained: high type error (model invents own categories)
  - MAVEN constrained: large type accuracy jump (flat labels easy to constrain)
  - WikiEvents constrained: smaller jump (harder hierarchy, 19.4% ignore constraint)
  - Both datasets: multi-event ambiguity drives wrong trigger errors
""")
