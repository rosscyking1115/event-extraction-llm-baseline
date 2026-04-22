import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"

messages = [
    {"role": "system", "content": "You extract event triggers and event types."},
    {
        "role": "user",
        "content": """You are an event extraction system.

Identify the main event trigger and event type in the sentence below.

Sentence: "The army attacked the city at dawn."

Output only valid JSON:
{"trigger": "...", "type": "..."}"""
    }
]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=80)

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
