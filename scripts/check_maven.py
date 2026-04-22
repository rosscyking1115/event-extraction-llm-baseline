import json
from pathlib import Path

base = Path(f"/mnt/parscratch/users/{Path.home().name}/team-rg1/data")

print("Searching for JSONL files under:", base)
jsonl_files = list(base.rglob("*.jsonl"))

print(f"Found {len(jsonl_files)} JSONL files")
for p in jsonl_files[:20]:
    print(p)

if not jsonl_files:
    raise SystemExit("No JSONL files found. Check your MAVEN path.")

path = jsonl_files[0]
print("\nOpening first JSONL file:", path)

with open(path, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()

obj = json.loads(first_line)

print("Top-level type of one line:", type(obj))

if isinstance(obj, dict):
    print("Keys:", list(obj.keys()))
    print("\nPreview:")
    for k, v in obj.items():
        print(f"{k}: {str(v)[:500]}")
else:
    print("Unexpected structure:", obj)
