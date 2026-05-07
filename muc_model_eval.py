#!/usr/bin/env python3
"""
muc_model_eval.py — Unified MUC-4 / MUC-6 LLM inference script.

Runs any HuggingFace causal LM on the parsed MUC datasets and saves
predictions as JSONL, ready for evaluate_muc.py.

Supports:
  - MUC-4 terrorism template filling (23 event slots)
  - MUC-6 corporate succession extraction (list of event dicts)
  - Zero-shot and few-shot prompting
  - Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (and others)

Usage examples:
    # MUC-4, zero-shot, Qwen
    python muc_model_eval.py \\
        --dataset muc4 \\
        --data_file /mnt/parscratch/users/acp25ck/team-rg1/data/muc4_tst3.json \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --prompt_type zero_shot \\
        --output_dir /mnt/parscratch/users/acp25ck/team-rg1/results/

    # MUC-4, few-shot (uses first N docs from dev split as examples)
    python muc_model_eval.py \\
        --dataset muc4 \\
        --data_file /mnt/parscratch/users/acp25ck/team-rg1/data/muc4_tst3.json \\
        --few_shot_file /mnt/parscratch/users/acp25ck/team-rg1/data/muc4_dev.json \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --prompt_type few_shot \\
        --n_few_shot 2 \\
        --output_dir /mnt/parscratch/users/acp25ck/team-rg1/results/

    # MUC-6, zero-shot, Llama
    python muc_model_eval.py \\
        --dataset muc6 \\
        --data_file /mnt/parscratch/users/acp25ck/team-rg1/data/muc6_test.json \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --prompt_type zero_shot \\
        --output_dir /mnt/parscratch/users/acp25ck/team-rg1/results/

Output JSONL format (one line per document):
    {"doc_id": "TST3-MUC4-0001", "prediction": {"INCIDENT_TYPE": "attack", ...}}
    {"doc_id": "9301060123",      "prediction": [{"succession_org": "...", ...}]}
"""

import os
import re
import json
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MUC-4/6 LLM inference for slot-filling evaluation"
    )
    p.add_argument("--dataset", required=True, choices=["muc4", "muc6"],
                   help="Which MUC dataset to evaluate")
    p.add_argument("--data_file", required=True,
                   help="Path to parsed JSON file (from parse_muc34.py or parse_muc6.py)")
    p.add_argument("--model", required=True,
                   help="HuggingFace model name or path")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save prediction JSONL")
    p.add_argument("--prompt_type", default="zero_shot",
                   choices=["zero_shot", "few_shot"],
                   help="Prompting strategy")
    p.add_argument("--few_shot_file", default=None,
                   help="JSON file with few-shot examples (different split from data_file)")
    p.add_argument("--n_few_shot", type=int, default=2,
                   help="Number of few-shot examples to include")
    p.add_argument("--max_docs", type=int, default=None,
                   help="Maximum documents to evaluate (for debugging)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens to generate per document")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Inference batch size (keep at 1 for long outputs)")
    p.add_argument("--skip_no_event", action="store_true",
                   help="Skip documents that have no gold event (for speed)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json_object(text):
    """
    Try to extract and parse a JSON object {} from model output.
    Returns parsed dict or None.
    """
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first { ... } block
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Try finding a longer JSON block (nested)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def extract_json_array(text):
    """
    Try to extract and parse a JSON array [...] from model output.
    Returns parsed list or None.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]  # single event returned as dict
    except Exception:
        pass
    # Find first [ ... ] block
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            pass
    # Try longer array block
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            pass
    # Model might have returned a single object when we expected a list
    obj = extract_json_object(text)
    if obj is not None:
        return [obj]
    return None


def extract_prediction(text, dataset):
    """Extract prediction from raw model output based on dataset type."""
    if dataset == "muc4":
        return extract_json_object(text)
    else:
        return extract_json_array(text)


# ---------------------------------------------------------------------------
# MUC-4 prompts
# ---------------------------------------------------------------------------

MUC4_SLOT_DESCRIPTIONS = """\
{
  "INCIDENT_TYPE": "Type of terrorist incident. One of: ATTACK, BOMBING, KIDNAPPING, ARSON, ASSASSINATION, ROBBERY, FORCED WORK STOPPAGE. Null if no incident.",
  "INCIDENT_DATE": "Date of the incident (as it appears in text). Null if not mentioned.",
  "INCIDENT_LOCATION": "Location where the incident occurred (country, city, or region). Null if not mentioned.",
  "INCIDENT_STAGE": "Stage of execution: ACCOMPLISHED (carried out) or ATTEMPTED (failed/foiled). Null if unclear.",
  "INCIDENT_INSTRUMENT_ID": "Specific weapon or instrument used (e.g. 'car bomb', 'pistol'). Null if none.",
  "INCIDENT_INSTRUMENT_TYPE": "General type of weapon (e.g. BOMB, GUN, GRENADE). Null if none.",
  "PERP_INCIDENT_CATEGORY": "Category of perpetrator motivation: TERRORIST ACT, STATE-SPONSORED VIOLENCE, POLITICAL VIOLENCE, CRIMINAL ACT. Null if unclear.",
  "PERP_INDIVIDUAL_ID": "Name or description of individual perpetrator(s) (e.g. 'armed men', 'John Doe'). Null if unknown.",
  "PERP_ORGANIZATION_ID": "Name of the perpetrating organization (e.g. 'Shining Path', 'FMLN'). Null if unknown.",
  "PERP_ORG_CONFIDENCE": "Confidence level: SUSPECTED OR ACCUSED, SUSPECTED OR ACCUSED BY AUTHORITIES, CONFIRMED. Null if no org identified.",
  "PHYS_TGT_ID": "Name or description of physical target (e.g. 'US Embassy', 'power station'). Null if none.",
  "PHYS_TGT_TYPE": "Type of physical target (e.g. GOVERNMENT BUILDING, VEHICLE, UTILITY). Null if none.",
  "PHYS_TGT_NUMBER": "Number of physical targets destroyed/damaged. Null if none.",
  "PHYS_TGT_FOREIGN_NATION": "Foreign nation associated with physical target. Null if none.",
  "PHYS_TGT_EFFECT": "Effect on physical target: DESTROYED, DAMAGED, DISRUPTED. Null if none.",
  "PHYS_TGT_TOTAL_NUMBER": "Total number of physical targets affected. Null if none.",
  "HUM_TGT_NAME": "Full name of specific human target(s). Null if not named.",
  "HUM_TGT_DESCRIPTION": "Description or role of human target(s) (e.g. 'businessman', 'police officer'). Null if none.",
  "HUM_TGT_TYPE": "Category of human target: CIVILIAN, GOVERNMENT OFFICIAL, MILITARY, POLICE, FORMER GOVERNMENT OFFICIAL. Null if none.",
  "HUM_TGT_NUMBER": "Number of human targets. Null if none.",
  "HUM_TGT_FOREIGN_NATION": "Foreign nation associated with human target. Null if none.",
  "HUM_TGT_EFFECT": "Effect on human targets: DEATH, INJURY, KIDNAPPING, NO INJURY. Null if none.",
  "HUM_TGT_TOTAL_NUMBER": "Total count of all human targets affected. Null if not given."
}"""

MUC4_EMPTY_TEMPLATE = {slot: None for slot in [
    "INCIDENT_TYPE", "INCIDENT_DATE", "INCIDENT_LOCATION", "INCIDENT_STAGE",
    "INCIDENT_INSTRUMENT_ID", "INCIDENT_INSTRUMENT_TYPE",
    "PERP_INCIDENT_CATEGORY", "PERP_INDIVIDUAL_ID",
    "PERP_ORGANIZATION_ID", "PERP_ORG_CONFIDENCE",
    "PHYS_TGT_ID", "PHYS_TGT_TYPE", "PHYS_TGT_NUMBER",
    "PHYS_TGT_FOREIGN_NATION", "PHYS_TGT_EFFECT", "PHYS_TGT_TOTAL_NUMBER",
    "HUM_TGT_NAME", "HUM_TGT_DESCRIPTION", "HUM_TGT_TYPE",
    "HUM_TGT_NUMBER", "HUM_TGT_FOREIGN_NATION",
    "HUM_TGT_EFFECT", "HUM_TGT_TOTAL_NUMBER",
]}


def build_muc4_example(doc):
    """Build a few-shot example string for MUC-4 from a gold document."""
    event_templates = [t for t in doc.get('templates', [])
                       if t.get('MESSAGE_TEMPLATE', '*') not in ('*', None)]
    if event_templates:
        tmpl = event_templates[0]
        pred = {slot: tmpl.get(slot) for slot in MUC4_EMPTY_TEMPLATE}
    else:
        pred = MUC4_EMPTY_TEMPLATE.copy()
    return doc['text'][:1500], json.dumps(pred, indent=2)


def build_muc4_prompt(doc_text, few_shot_examples=None):
    """
    Construct the MUC-4 zero-shot or few-shot prompt.
    few_shot_examples: list of (article_text, json_str) tuples
    """
    system = (
        "You are an expert information extraction system specialised in terrorism event analysis. "
        "Read the news article carefully and fill in the MUC-4 terrorism template. "
        "If the article does not describe a relevant terrorism event, set all fields to null. "
        "Output ONLY a valid JSON object — no explanation, no markdown, no extra text."
    )

    slot_schema = (
        "Fill in ALL of the following fields (use null if not present in the article):\n"
        + MUC4_SLOT_DESCRIPTIONS
    )

    examples_str = ""
    if few_shot_examples:
        examples_str = "\n\nHere are examples of correctly filled templates:\n"
        for i, (ex_text, ex_json) in enumerate(few_shot_examples, 1):
            examples_str += f"\nEXAMPLE {i}:\nArticle:\n{ex_text[:1200]}\n\nOutput:\n{ex_json}\n"
        examples_str += "\nNow fill in the template for the following article:\n"

    prompt = (
        f"{system}\n\n"
        f"{slot_schema}"
        f"{examples_str}\n\n"
        f"Article:\n{doc_text[:3000]}\n\n"
        f"JSON output:"
    )
    return prompt


# ---------------------------------------------------------------------------
# MUC-6 prompts
# ---------------------------------------------------------------------------

MUC6_SLOT_DESCRIPTIONS = """\
Each succession event should be a JSON object with these fields:
{
  "succession_org": "Name of the organisation where the succession occurs. Null if unclear.",
  "post": "The job title or position (e.g. 'chief executive', 'president', 'chairman'). Null if not specified.",
  "person_in": "Full name of the person TAKING the position (new hire/promotion). Null if no incoming person.",
  "person_out": "Full name of the person LEAVING the position (resigned/retired/fired). Null if no outgoing person.",
  "vacancy_reason": "Why the position is vacated: RETIREMENT, FIRED, RESIGNED, REASSIGNMENT, NEW_POST_CREATED, OTH_UNK. Null if unknown.",
  "on_the_job_in": "Has the incoming person already started? YES, NO, or UNCLEAR. Null if no incoming person.",
  "on_the_job_out": "Has the outgoing person already left? YES, NO, or UNCLEAR. Null if no outgoing person.",
  "other_org_in": "If the incoming person comes from another organisation, name it. Null otherwise.",
  "rel_other_org_in": "Relationship to other_org_in: SAME_ORG, RELATED_ORG, or OTHER_ORG. Null if no other_org."
}"""


def build_muc6_example(doc):
    """Build a few-shot example string for MUC-6 from a gold document."""
    events = doc.get('succession_events', [])
    return doc['text'][:1500], json.dumps(events, indent=2)


def build_muc6_prompt(doc_text, few_shot_examples=None):
    """
    Construct the MUC-6 zero-shot or few-shot prompt.
    """
    system = (
        "You are an expert information extraction system specialised in corporate news analysis. "
        "Read the Wall Street Journal article and identify ALL executive succession events "
        "(people being hired, fired, retiring, or changing roles at organisations). "
        "Output a JSON array of succession events. "
        "If there are no succession events, output an empty array: []. "
        "Output ONLY the JSON array — no explanation, no markdown, no extra text."
    )

    slot_schema = MUC6_SLOT_DESCRIPTIONS

    examples_str = ""
    if few_shot_examples:
        examples_str = "\n\nHere are examples of correctly extracted succession events:\n"
        for i, (ex_text, ex_json) in enumerate(few_shot_examples, 1):
            examples_str += f"\nEXAMPLE {i}:\nArticle:\n{ex_text[:1200]}\n\nOutput:\n{ex_json}\n"
        examples_str += "\nNow extract succession events from the following article:\n"

    prompt = (
        f"{system}\n\n"
        f"{slot_schema}"
        f"{examples_str}\n\n"
        f"Article:\n{doc_text[:3000]}\n\n"
        f"JSON output:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Few-shot example loading
# ---------------------------------------------------------------------------

def load_few_shot_examples(few_shot_file, dataset, n):
    """
    Load the first N documents with events from a JSON file to use as examples.
    Returns list of (article_text, json_str) tuples.
    """
    with open(few_shot_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for doc in data:
        if not doc.get('has_event', False):
            continue
        if dataset == "muc4":
            text, json_str = build_muc4_example(doc)
        else:
            text, json_str = build_muc6_example(doc)
        examples.append((text, json_str))
        if len(examples) >= n:
            break

    return examples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name):
    """Load tokenizer and model in float16 on GPU."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(prompt, tokenizer, model, max_new_tokens):
    """Run model inference on a single prompt, return decoded output string."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"{'='*60}")
    print(f"  MUC Model Evaluation")
    print(f"  Dataset:      {args.dataset.upper()}")
    print(f"  Model:        {args.model}")
    print(f"  Prompt type:  {args.prompt_type}")
    print(f"  Data file:    {args.data_file}")
    print(f"{'='*60}\n")

    # Load dataset
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.max_docs:
        data = data[:args.max_docs]

    if args.skip_no_event:
        before = len(data)
        data = [d for d in data if d.get('has_event', True)]
        print(f"  Skipping docs without events: {before} -> {len(data)}")

    print(f"  {len(data)} documents to evaluate\n")

    # Load few-shot examples if needed
    few_shot_examples = None
    if args.prompt_type == "few_shot":
        src = args.few_shot_file or args.data_file
        print(f"Loading {args.n_few_shot} few-shot examples from {src}...")
        few_shot_examples = load_few_shot_examples(src, args.dataset, args.n_few_shot)
        print(f"  Loaded {len(few_shot_examples)} examples\n")

    # Output path
    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.replace('/', '_').replace('-', '_').lower()
    split_name = Path(args.data_file).stem  # e.g. muc4_tst3
    out_file = os.path.join(
        args.output_dir,
        f"{split_name}_{model_slug}_{args.prompt_type}.jsonl"
    )
    print(f"Output file: {out_file}\n")

    # Load model
    tokenizer, model = load_model(args.model)

    # Inference loop
    results = []
    n_valid_json = 0
    start_time = time.time()

    for i, doc in enumerate(data):
        doc_id = doc['doc_id']
        text = doc.get('text', '')

        # Build prompt
        if args.dataset == "muc4":
            prompt = build_muc4_prompt(text, few_shot_examples)
        else:
            prompt = build_muc6_prompt(text, few_shot_examples)

        # Run inference
        raw_output = run_inference(prompt, tokenizer, model, args.max_new_tokens)

        # Extract prediction
        prediction = extract_prediction(raw_output, args.dataset)
        is_valid = prediction is not None

        if is_valid:
            n_valid_json += 1

        # Build result record
        result = {
            "doc_id": doc_id,
            "prediction": prediction,
            "raw_output": raw_output[:500],  # truncate for storage
            "json_valid": is_valid,
            "model": args.model,
            "prompt_type": args.prompt_type,
        }
        results.append(result)

        # Progress logging
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(data) - i - 1)
        valid_rate = n_valid_json / (i + 1)

        print(f"[{i+1:4d}/{len(data)}] {doc_id}  "
              f"JSON_valid={int(is_valid)}  "
              f"rate={valid_rate:.2f}  "
              f"ETA={remaining/60:.1f}min")

        # Show prediction preview
        if is_valid and args.dataset == "muc4":
            itype = prediction.get('INCIDENT_TYPE', '?')
            iloc = prediction.get('INCIDENT_LOCATION', '?')
            print(f"           -> TYPE={itype}  LOC={iloc}")
        elif is_valid and args.dataset == "muc6":
            n_events = len(prediction) if isinstance(prediction, list) else 1
            print(f"           -> {n_events} succession event(s) extracted")

        # Save incrementally every 10 docs (crash protection)
        if (i + 1) % 10 == 0:
            _save_results(results, out_file)

    # Final save
    _save_results(results, out_file)

    # Summary
    n = len(results)
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  INFERENCE COMPLETE")
    print(f"  Model:          {args.model}")
    print(f"  Dataset:        {args.dataset.upper()}")
    print(f"  Documents:      {n}")
    print(f"  Valid JSON:     {n_valid_json}/{n} ({100*n_valid_json/n:.1f}%)")
    print(f"  Total time:     {elapsed/60:.1f} min")
    print(f"  Avg per doc:    {elapsed/n:.1f} sec")
    print(f"  Results saved:  {out_file}")
    print(f"{'='*60}\n")
    print(f"Next step — run evaluation:")
    print(f"  python evaluate_muc.py \\")
    print(f"    --gold {args.data_file} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --predictions {out_file} \\")
    print(f"    --model \"{args.model}\" \\")
    print(f"    --prompt_type {args.prompt_type} \\")
    print(f"    --output_csv {args.output_dir}/{split_name}_{model_slug}_{args.prompt_type}_scores.csv")


def _save_results(results, out_file):
    """Save results list to JSONL file (overwrite)."""
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
