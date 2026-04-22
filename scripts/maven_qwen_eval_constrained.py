import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE = Path("/mnt/parscratch/users/acp25ck/team-rg1/data/MAVEN Event Detection/train.jsonl")
RESULTS_FILE = Path("/mnt/parscratch/users/acp25ck/team-rg1/results/maven_qwen_eval_constrained_results.jsonl")


def load_maven_sentence_samples(path: Path, max_samples: int = 50):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            content = doc.get("content", [])
            events = doc.get("events", [])

            for event in events:
                event_type = event.get("type", "")
                mentions = event.get("mention", [])

                for mention in mentions:
                    sent_id = mention.get("sent_id")
                    trigger_word = mention.get("trigger_word", "")

                    if sent_id is None or sent_id >= len(content):
                        continue

                    sentence = content[sent_id]["sentence"]

                    samples.append({
                        "sentence": sentence,
                        "gold_trigger": trigger_word,
                        "gold_type": event_type
                    })

                    if len(samples) >= max_samples:
                        return samples

    return samples


def build_prompt(sentence: str, candidate_types):
    label_text = ", ".join(candidate_types)

    return f"""You are an event extraction system.

Task:
1. Identify the main event trigger in the sentence.
2. Choose the event type from the candidate list only.
3. Output English only.
4. Output valid JSON only.

Candidate event types:
[{label_text}]

Sentence:
"{sentence}"

Return exactly this format:
{{"trigger": "...", "type": "..."}}

Rules:
- The type must be one of the candidate event types exactly as written.
- Do not output Chinese.
- Do not invent a new label.
- If no event fits, output:
{{"trigger": "NONE", "type": "NONE"}}"""
    

def safe_parse_prediction(text: str):
    try:
        pred = json.loads(text)
        trigger = str(pred.get("trigger", "")).strip()
        event_type = str(pred.get("type", "")).strip()
        return {
            "valid_json": True,
            "pred_trigger": trigger,
            "pred_type": event_type
        }
    except Exception:
        return {
            "valid_json": False,
            "pred_trigger": "",
            "pred_type": ""
        }


def normalize_text(x: str) -> str:
    return x.strip().lower()


def run_model(tokenizer, model, sentence: str, candidate_types):
    prompt = build_prompt(sentence, candidate_types)

    messages = [
        {"role": "system", "content": "You extract event triggers and event types."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=80)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return result


def main():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    print("Loading MAVEN samples...")
    samples = load_maven_sentence_samples(TRAIN_FILE, max_samples=50)
    print(f"Loaded {len(samples)} samples")

    candidate_types = sorted(set(sample["gold_type"] for sample in samples))
    print("Candidate types:", candidate_types)

    valid_json_count = 0
    trigger_match_count = 0
    type_match_count = 0
    both_match_count = 0

    with open(RESULTS_FILE, "w", encoding="utf-8") as out_f:
        for i, sample in enumerate(samples, start=1):
            raw_prediction = run_model(tokenizer, model, sample["sentence"], candidate_types)
            parsed = safe_parse_prediction(raw_prediction)

            gold_trigger_norm = normalize_text(sample["gold_trigger"])
            gold_type_norm = normalize_text(sample["gold_type"])
            pred_trigger_norm = normalize_text(parsed["pred_trigger"])
            pred_type_norm = normalize_text(parsed["pred_type"])

            trigger_match = pred_trigger_norm == gold_trigger_norm
            type_match = pred_type_norm == gold_type_norm
            both_match = trigger_match and type_match

            if parsed["valid_json"]:
                valid_json_count += 1
            if trigger_match:
                trigger_match_count += 1
            if type_match:
                type_match_count += 1
            if both_match:
                both_match_count += 1

            record = {
                "id": i,
                "sentence": sample["sentence"],
                "gold_trigger": sample["gold_trigger"],
                "gold_type": sample["gold_type"],
                "candidate_types": candidate_types,
                "raw_prediction": raw_prediction,
                "valid_json": parsed["valid_json"],
                "pred_trigger": parsed["pred_trigger"],
                "pred_type": parsed["pred_type"],
                "trigger_match": trigger_match,
                "type_match": type_match,
                "both_match": both_match
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("\n" + "=" * 80)
            print(f"Sample {i}")
            print("Sentence      :", sample["sentence"])
            print("Gold trigger  :", sample["gold_trigger"])
            print("Gold type     :", sample["gold_type"])
            print("Prediction    :", raw_prediction)
            print("Valid JSON    :", parsed["valid_json"])
            print("Trigger match :", trigger_match)
            print("Type match    :", type_match)
            print("Both match    :", both_match)

    total = len(samples)
    print("\n" + "#" * 80)
    print("FINAL RESULTS")
    print(f"Total samples              : {total}")
    print(f"Valid JSON rate            : {valid_json_count}/{total} = {valid_json_count/total:.3f}")
    print(f"Trigger exact match acc    : {trigger_match_count}/{total} = {trigger_match_count/total:.3f}")
    print(f"Type exact match acc       : {type_match_count}/{total} = {type_match_count/total:.3f}")
    print(f"Trigger+Type exact acc     : {both_match_count}/{total} = {both_match_count/total:.3f}")
    print(f"Saved detailed results to  : {RESULTS_FILE}")


if __name__ == "__main__":
    main()
