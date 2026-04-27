"""
wikievents_rule_baseline.py
Simple rule-based majority-class baseline for WikiEvents.
Two strategies:
  1. majority_type: always predict the most frequent event type from training data
  2. first_verb: predict the first verb in the sentence as trigger, majority type for type

This establishes a lower-bound floor to compare LLM results against.
Run from ~/team-rg1:
    python scripts/wikievents_rule_baseline.py
"""

import json
import os
import re
from collections import Counter

TRAIN_FILE = "/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/train.jsonl"
DATA_FILE  = "/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/dev.jsonl"
OUT_FILE   = "/mnt/parscratch/users/acp25ck/team-rg1/results/wikievents_rule_baseline_results.jsonl"

def get_sentence_text(doc, sent_idx):
    return doc['sentences'][sent_idx][1]

def levenshtein(s1, s2):
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2: return 0
    if not s1: return len(s2)
    if not s2: return len(s1)
    matrix = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
    for i in range(len(s1)+1): matrix[i][0] = i
    for j in range(len(s2)+1): matrix[0][j] = j
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+cost)
    return matrix[len(s1)][len(s2)]

def type_partial_credit(gold, pred):
    if gold == pred: return 1.0
    gp = gold.split('.'); pp = pred.split('.')
    if len(gp) < 2: return 0.0
    if gp[0]==pp[0] and len(pp)>1 and gp[1]==pp[1]: return 0.67
    if gp[0]==pp[0]: return 0.33
    return 0.0

# ── Build stats from training data ────────────────────────────────────────────
print("Analysing training data...")
type_counter    = Counter()
trigger_counter = Counter()

with open(TRAIN_FILE) as f:
    for line in f:
        doc = json.loads(line.strip())
        for em in doc['event_mentions']:
            type_counter[em['event_type']] += 1
            trigger_counter[em['trigger']['text'].lower()] += 1

majority_type    = type_counter.most_common(1)[0][0]
majority_trigger = trigger_counter.most_common(1)[0][0]

print(f"Majority type:    {majority_type} ({type_counter[majority_type]} occurrences)")
print(f"Majority trigger: {majority_trigger} ({trigger_counter[majority_trigger]} occurrences)")
print(f"Total event types seen: {len(type_counter)}\n")

# ── Load dev data ─────────────────────────────────────────────────────────────
docs = []
with open(DATA_FILE) as f:
    for line in f:
        docs.append(json.loads(line.strip()))

samples = []
for doc in docs:
    for em in doc['event_mentions']:
        sent_idx = em['trigger']['sent_idx']
        try:
            sent_text = get_sentence_text(doc, sent_idx)
        except (IndexError, KeyError):
            continue
        samples.append({
            'doc_id':       doc['doc_id'],
            'sentence':     sent_text,
            'gold_trigger': em['trigger']['text'].lower().strip(),
            'gold_type':    em['event_type'].strip(),
        })

print(f"Evaluating on {len(samples)} event mentions from {len(docs)} documents.\n")

# ── Evaluate two strategies ───────────────────────────────────────────────────
strategies = {
    'majority_type_only': {
        'pred_trigger': majority_trigger,
        'pred_type':    majority_type,
    },
}

results_all = {}

for strategy_name, pred_defaults in strategies.items():
    results = []
    trigger_correct = type_correct = both_correct = 0
    trigger_lev_sum = type_partial_sum = 0.0

    for sample in samples:
        pred_trigger = pred_defaults['pred_trigger']
        pred_type    = pred_defaults['pred_type']

        t_exact = pred_trigger == sample['gold_trigger']
        y_exact = pred_type    == sample['gold_type']
        b_exact = t_exact and y_exact

        t_lev  = 1.0 - levenshtein(pred_trigger, sample['gold_trigger']) / max(len(pred_trigger), len(sample['gold_trigger']), 1)
        y_part = type_partial_credit(sample['gold_type'], pred_type)

        if t_exact: trigger_correct += 1
        if y_exact: type_correct    += 1
        if b_exact: both_correct    += 1
        trigger_lev_sum  += t_lev
        type_partial_sum += y_part

        results.append({
            'strategy':        strategy_name,
            'doc_id':          sample['doc_id'],
            'sentence':        sample['sentence'][:200],
            'gold_trigger':    sample['gold_trigger'],
            'gold_type':       sample['gold_type'],
            'pred_trigger':    pred_trigger,
            'pred_type':       pred_type,
            'valid_json':      True,
            'trigger_correct': t_exact,
            'type_correct':    y_exact,
            'both_correct':    b_exact,
        })

    n = len(results)
    results_all[strategy_name] = results

    print(f"Strategy: {strategy_name}")
    print(f"  Predicted trigger: '{pred_defaults['pred_trigger']}'")
    print(f"  Predicted type:    '{pred_defaults['pred_type']}'")
    print(f"  Trigger exact acc: {trigger_correct}/{n} = {trigger_correct/n:.3f}")
    print(f"  Type exact acc:    {type_correct}/{n} = {type_correct/n:.3f}")
    print(f"  Both exact acc:    {both_correct}/{n} = {both_correct/n:.3f}")
    print(f"  Avg trigger Lev:   {trigger_lev_sum/n:.3f}")
    print(f"  Type partial cred: {type_partial_sum/n:.3f}")
    print()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, 'w') as f:
    for strategy_results in results_all.values():
        for r in strategy_results:
            f.write(json.dumps(r) + '\n')

print(f"Results saved to: {OUT_FILE}")
print("\nNote: These rule-based baselines establish a lower-bound floor.")
print("Any LLM experiment scoring above these numbers adds genuine value.")
