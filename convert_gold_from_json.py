#!/usr/bin/env python3
"""
Convert Muc34.json (tst3.json) to gold_labels.json for eval.py.
Applies xref stripping and preserves OR alternatives.
"""
import json
import re
import sys

NULL = "-"

def strip_xref(value: str) -> str:
    """
    Strip xref notation from each OR branch.
    e.g.  CIVILIAN: "JESUIT PRIESTS"  → CIVILIAN
          DEATH: "JESUIT PRIESTS"     → DEATH
          "CAR BOMB": "INSTRUMENT"    → "CAR BOMB"
    Also strips standalone colon-references like:
          "GOVERNOR": "ANTONIO ROLDAN BETANCUR" → "GOVERNOR"
    """
    branches = re.split(r"\s*/\s*", value)
    cleaned = []
    for branch in branches:
        branch = branch.strip()
        # Strip everything from ': "' onwards (xref pattern)
        m = re.match(r'^(.+?)\s*:\s*".*$', branch)
        if m:
            branch = m.group(1).strip()
        cleaned.append(branch)
    return " / ".join(cleaned)


def clean_value(raw: str) -> str:
    """Strip xref notation, normalise whitespace, return NULL for empties."""
    if not raw or raw.strip() in {"-", "?", ""}:
        return NULL
    v = strip_xref(raw.strip())
    if not v or v in {"-", "?"}:
        return NULL
    return v


def convert(src_path: str, dst_path: str):
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    gold = {}
    n_docs = n_templates = 0

    for record in data["records"]:
        doc_id = record["message_id"]
        templates_out = []

        for tmpl in record.get("templates", []):
            row = {str(i): NULL for i in range(25)}
            row["0"] = doc_id

            for slot in tmpl.get("slots", []):
                idx = slot["index"]
                raw = slot.get("raw_value", NULL)
                v = clean_value(raw)
                row[str(idx)] = v

            templates_out.append(row)

        if templates_out:
            gold[doc_id] = templates_out
            n_docs += 1
            n_templates += len(templates_out)

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    print(f"Converted {n_docs} documents, {n_templates} templates → {dst_path}")
    # Sanity checks
    first_id = next(iter(gold))
    print(f"\nSample ({first_id}, template 1):")
    for k in sorted(gold[first_id][0].keys(), key=int):
        v = gold[first_id][0][k]
        if v != "-":
            print(f"  {int(k):>2}: {v}")

    # Check a known xref slot
    test_id = "TST3-MUC4-0001"
    if test_id in gold:
        print(f"\nXref check ({test_id} slot 20 — should be 'CIVILIAN'):")
        print(f"  {gold[test_id][0]['20']}")

    # Check a known OR alternative
    test_id2 = "TST3-MUC4-0002"
    if test_id2 in gold:
        print(f"\nOR-alternatives check ({test_id2} slot 9 — should show multiple / values):")
        print(f"  {gold[test_id2][0]['9']}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "tst3.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "gold_labels_tst3.json"
    convert(src, dst)
