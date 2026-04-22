import json
from pathlib import Path
from collections import Counter

V1_FILE = Path("/mnt/parscratch/users/acp25ck/team-rg1/results/maven_qwen_eval_results.jsonl")
V2_FILE = Path("/mnt/parscratch/users/acp25ck/team-rg1/results/maven_qwen_eval_constrained_results.jsonl")


def normalize(x: str) -> str:
    return str(x).strip().lower()


def classify_error(record):
    gold_trigger = normalize(record["gold_trigger"])
    gold_type = normalize(record["gold_type"])
    pred_trigger = normalize(record["pred_trigger"])
    pred_type = normalize(record["pred_type"])

    if record["both_match"]:
        return "correct"

    if pred_trigger == gold_trigger and pred_type != gold_type:
        return "right trigger, wrong type"

    if pred_trigger != gold_trigger and pred_type == gold_type:
        return "wrong trigger, right type"

    # likely ontology mismatch if prediction is event-like but type differs
    if pred_trigger and pred_type and pred_type != gold_type:
        return "ontology/type mismatch"

    return "other"


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def summarize(path, label):
    records = load_records(path)
    counts = Counter(classify_error(r) for r in records)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print(f"Total records: {len(records)}")
    for k, v in counts.items():
        print(f"{k}: {v}")

    print("\nExamples:")
    shown = 0
    for r in records:
        category = classify_error(r)
        if category != "correct":
            print("-" * 80)
            print("Category     :", category)
            print("Sentence     :", r["sentence"])
            print("Gold trigger :", r["gold_trigger"])
            print("Gold type    :", r["gold_type"])
            print("Pred trigger :", r["pred_trigger"])
            print("Pred type    :", r["pred_type"])
            shown += 1
        if shown >= 10:
            break


def main():
    summarize(V1_FILE, "V1: unconstrained zero-shot")
    summarize(V2_FILE, "V2: constrained-label zero-shot")


if __name__ == "__main__":
    main()
