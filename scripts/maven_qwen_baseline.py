import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE = Path("/mnt/parscratch/users/acp25ck/team-rg1/data/MAVEN Event Detection/train.jsonl")


def build_prompt(sentence: str) -> str:
    return f"""You are an event extraction system.

Identify the main event trigger and event type in the sentence below.

Sentence: "{sentence}"

Output only valid JSON:
{{"trigger": "...", "type": "..."}}

If no event is present, output:
{{"trigger": "NONE", "type": "NONE"}}"""


def load_maven_sentence_samples(path: Path, max_samples: int = 5):
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

                    if sent_id is None:
                        continue
                    if sent_id >= len(content):
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


def run_model(tokenizer, model, sentence: str):
    prompt = build_prompt(sentence)

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
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    print("Loading MAVEN samples...")
    samples = load_maven_sentence_samples(TRAIN_FILE, max_samples=5)

    print(f"Loaded {len(samples)} samples")

    for i, sample in enumerate(samples, start=1):
        prediction = run_model(tokenizer, model, sample["sentence"])

        print("\n" + "=" * 80)
        print(f"Sample {i}")
        print("Sentence     :", sample["sentence"])
        print("Gold trigger :", sample["gold_trigger"])
        print("Gold type    :", sample["gold_type"])
        print("Prediction   :", prediction)


if __name__ == "__main__":
    main()
