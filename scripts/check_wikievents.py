import json
from collections import Counter

DATA_FILE = '/mnt/parscratch/users/acp25ck/team-rg1/data/wikievents/train.jsonl'

def get_sentence_text(doc, sent_idx):
    # sentences[sent_idx] = [list_of_tokens, sentence_string]
    return doc['sentences'][sent_idx][1]

print("Loading WikiEvents train split...")
docs = []
with open(DATA_FILE) as f:
    for line in f:
        docs.append(json.loads(line.strip()))

print(f"Total documents: {len(docs)}")

total_events = sum(len(d['event_mentions']) for d in docs)
total_args = sum(len(em['arguments']) for d in docs for em in d['event_mentions'])
events_with_args = sum(1 for d in docs for em in d['event_mentions'] if em['arguments'])

print(f"Total event mentions: {total_events}")
print(f"Event mentions with arguments: {events_with_args} ({events_with_args/total_events:.1%})")
print(f"Total arguments: {total_args}")
print(f"Avg arguments per event (when present): {total_args/events_with_args:.2f}")

# Event type distribution
type_counter = Counter()
top_level_counter = Counter()
mid_level_counter = Counter()
for d in docs:
    for em in d['event_mentions']:
        type_counter[em['event_type']] += 1
        parts = em['event_type'].split('.')
        top_level_counter[parts[0]] += 1
        if len(parts) >= 2:
            mid_level_counter[f"{parts[0]}.{parts[1]}"] += 1

print(f"\nUnique full event types: {len(type_counter)}")
print(f"Unique mid-level types (Cat.Sub): {len(mid_level_counter)}")
print(f"Top-level categories: {len(top_level_counter)}")

print("\nTop-level category counts:")
for cat, count in top_level_counter.most_common():
    print(f"  {cat}: {count}")

print("\nMost common full event types (top 15):")
for t, count in type_counter.most_common(15):
    print(f"  {t}: {count}")

print("\n--- Sample event mentions ---")
shown = 0
for d in docs[:5]:
    for em in d['event_mentions']:
        if shown >= 5:
            break
        sent_idx = em['trigger']['sent_idx']
        try:
            sent_text = get_sentence_text(d, sent_idx)
        except (IndexError, KeyError):
            sent_text = "[could not retrieve sentence]"
        print(f"\nDoc: {d['doc_id']}")
        print(f"Sentence: {sent_text[:200]}")
        print(f"Trigger: '{em['trigger']['text']}'")
        print(f"Event type: {em['event_type']}")
        if em['arguments']:
            print(f"Arguments ({len(em['arguments'])}):")
            for arg in em['arguments'][:3]:
                print(f"  role={arg['role']}, text='{arg['text']}'")
        shown += 1
    if shown >= 5:
        break
