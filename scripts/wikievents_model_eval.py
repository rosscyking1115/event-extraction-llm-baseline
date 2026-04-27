"""
wikievents_model_eval.py
Reusable constrained zero-shot eval script for any HuggingFace model.
Used for model comparison experiments (e.g. Qwen2.5-14B, Llama-3.1-8B).

Usage:
    python scripts/wikievents_model_eval.py --model Qwen/Qwen2.5-14B-Instruct
    python scripts/wikievents_model_eval.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import json
import re
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
parser.add_argument('--max_docs', type=int, default=50)
parser.add_argument('--max_new_tokens', type=int, default=100)
args = parser.parse_args()

MODEL_NAME = args.model
TRAIN_FILE = "/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/train.jsonl"
DATA_FILE  = "/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/dev.jsonl"

# Derive output filename from model name
model_slug = MODEL_NAME.replace('/', '_').replace('-', '_').lower()
OUT_FILE = f"/mnt/parscratch/users/acp25ck/team-rg1/results/wikievents_{model_slug}_constrained_results.jsonl"

print(f"Model:      {MODEL_NAME}")
print(f"Output:     {OUT_FILE}")
print(f"Max docs:   {args.max_docs}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sentence_text(doc, sent_idx):
    return doc['sentences'][sent_idx][1]

def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None

# ── Build label set ───────────────────────────────────────────────────────────
print("Building label set from training split...")
label_set = set()
with open(TRAIN_FILE) as f:
    for line in f:
        doc = json.loads(line.strip())
        for em in doc['event_mentions']:
            label_set.add(em['event_type'])
label_list = sorted(label_set)
label_str  = "\n".join(f"  {t}" for t in label_list)
print(f"Found {len(label_list)} unique event types.\n")

def build_prompt(sentence: str) -> str:
    return (
        "You are an event extraction system.\n"
        "Given a sentence, identify the main event trigger word and classify its event type.\n\n"
        "You MUST choose the event_type from ONLY this list:\n"
        f"{label_str}\n\n"
        f"Sentence: {sentence}\n\n"
        "Respond with ONLY a valid JSON object with exactly two fields:\n"
        '  {"trigger": "<trigger word>", "event_type": "<must be from the list above>"}\n\n'
        "JSON:"
    )

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded.\n")

# ── Load data ─────────────────────────────────────────────────────────────────
docs = []
with open(DATA_FILE) as f:
    for line in f:
        docs.append(json.loads(line.strip()))
docs = docs[:args.max_docs]

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

print(f"Documents: {len(docs)} -> Event mentions: {len(samples)}\n")

# ── Inference ─────────────────────────────────────────────────────────────────
results = []
valid_json = trigger_correct = type_correct = both_correct = 0
invalid_type_count = 0

for i, sample in enumerate(samples):
    prompt = build_prompt(sample['sentence'])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed = extract_json(raw_output)
    is_valid = parsed is not None and 'trigger' in parsed and 'event_type' in parsed

    pred_trigger = parsed['trigger'].lower().strip() if is_valid else ""
    pred_type    = parsed['event_type'].strip()      if is_valid else ""

    type_in_set = pred_type in label_set
    if is_valid and not type_in_set:
        invalid_type_count += 1

    t_correct = pred_trigger == sample['gold_trigger']
    y_correct = pred_type    == sample['gold_type']
    b_correct = t_correct and y_correct

    if is_valid:  valid_json += 1
    if t_correct: trigger_correct += 1
    if y_correct: type_correct += 1
    if b_correct: both_correct += 1

    results.append({
        'model':           MODEL_NAME,
        'doc_id':          sample['doc_id'],
        'sentence':        sample['sentence'][:200],
        'gold_trigger':    sample['gold_trigger'],
        'gold_type':       sample['gold_type'],
        'pred_trigger':    pred_trigger,
        'pred_type':       pred_type,
        'raw_output':      raw_output,
        'valid_json':      is_valid,
        'type_in_set':     type_in_set,
        'trigger_correct': t_correct,
        'type_correct':    y_correct,
        'both_correct':    b_correct,
    })

    n = i + 1
    print(f"[{n:4d}/{len(samples)}]  gold=('{sample['gold_trigger']}', {sample['gold_type']})  pred=('{pred_trigger}', {pred_type})  T={int(t_correct)} Y={int(y_correct)}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

n = len(results)
print("\n" + "="*60)
print(f"WIKIEVENTS CONSTRAINED -- {MODEL_NAME}")
print("="*60)
print(f"Documents evaluated:        {len(docs)}")
print(f"Event mentions evaluated:   {n}")
print(f"Valid JSON rate:             {valid_json}/{n} = {valid_json/n:.3f}")
print(f"Types within label set:     {valid_json - invalid_type_count}/{valid_json}")
print(f"Trigger exact match acc:    {trigger_correct}/{n} = {trigger_correct/n:.3f}")
print(f"Type exact match acc:       {type_correct}/{n} = {type_correct/n:.3f}")
print(f"Trigger+Type exact match:   {both_correct}/{n} = {both_correct/n:.3f}")
print(f"\nResults saved to: {OUT_FILE}")
