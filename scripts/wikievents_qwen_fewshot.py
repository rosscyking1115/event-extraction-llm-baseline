"""
wikievents_qwen_fewshot.py
Few-shot (3-example) constrained baseline using Qwen2.5-7B-Instruct on WikiEvents.
Examples are drawn from the training split, covering three different event categories.
Run from ~/team-rg1:
    python scripts/wikievents_qwen_fewshot.py
"""

import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE = "/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/train.jsonl"
DATA_FILE  = "/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/dev.jsonl"
OUT_FILE   = "/mnt/parscratch/users/acp25ck/team-rg1/results/wikievents_qwen_fewshot_results.jsonl"
MAX_DOCS   = 50
MAX_NEW_TOKENS = 100

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

# ── Build label set from training data ───────────────────────────────────────
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

# ── Few-shot examples (hand-picked from training split for clarity/diversity) ─
# Covers: Conflict, Life, Contact — three distinct top-level categories
FEW_SHOT_EXAMPLES = [
    {
        "sentence": "A suicide bomber detonated an explosive vest near the entrance of the government building, killing three guards.",
        "trigger":  "detonated",
        "event_type": "Conflict.Attack.DetonateExplode"
    },
    {
        "sentence": "The soldier was fatally shot during the ambush near the northern border.",
        "trigger":  "shot",
        "event_type": "Life.Die.Unspecified"
    },
    {
        "sentence": "The president addressed the nation in a live television broadcast on Friday evening.",
        "trigger":  "addressed",
        "event_type": "Contact.Contact.Broadcast"
    },
]

def format_example(ex):
    return (
        f'Sentence: {ex["sentence"]}\n'
        f'JSON: {{"trigger": "{ex["trigger"]}", "event_type": "{ex["event_type"]}"}}'
    )

EXAMPLES_BLOCK = "\n\n".join(format_example(e) for e in FEW_SHOT_EXAMPLES)

def build_prompt(sentence: str) -> str:
    return (
        "You are an event extraction system.\n"
        "Given a sentence, identify the main event trigger word and classify its event type.\n\n"
        "You MUST choose the event_type from ONLY this list:\n"
        f"{label_str}\n\n"
        "Here are three examples of correct extraction:\n\n"
        f"{EXAMPLES_BLOCK}\n\n"
        "Now extract the event from this sentence:\n"
        f"Sentence: {sentence}\n"
        'JSON: {"trigger":'
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
docs = docs[:MAX_DOCS]

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
    # Note: we prime the output with '{"trigger":' so we prepend it to decoded output
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Re-attach the primed prefix so JSON parsing works
    full_output = '{"trigger":' + raw_output

    parsed = extract_json(full_output)
    if parsed is None:
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
print("WIKIEVENTS FEW-SHOT (3-example) CONSTRAINED -- RESULTS")
print("="*60)
print(f"Documents evaluated:        {len(docs)}")
print(f"Event mentions evaluated:   {n}")
print(f"Valid JSON rate:             {valid_json}/{n} = {valid_json/n:.3f}")
print(f"Types within label set:     {valid_json - invalid_type_count}/{valid_json}")
print(f"Trigger exact match acc:    {trigger_correct}/{n} = {trigger_correct/n:.3f}")
print(f"Type exact match acc:       {type_correct}/{n} = {type_correct/n:.3f}")
print(f"Trigger+Type exact match:   {both_correct}/{n} = {both_correct/n:.3f}")
print(f"\nResults saved to: {OUT_FILE}")
