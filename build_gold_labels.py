#!/usr/bin/env python3
"""
build_gold_labels.py
====================
Converts a MUC-3/4 key template file (key-tst3.v2 or any .ans / response file
in the numbered-slot MUC-3 format) into gold_labels.json for use with eval.py.

The MUC-3 template format looks like:
    ; event "TST3-MUC4-0001".1234
    0.  MESSAGE: ID                    TST3-MUC4-0001
    1.  MESSAGE: TEMPLATE              1
    2.  INCIDENT: DATE                 -
    3.  INCIDENT: LOCATION             EL SALVADOR
    4.  INCIDENT: TYPE                 ATTACK / BOMBING        ← OR alternatives
    5.  INCIDENT: STAGE OF EXECUTION   ACCOMPLISHED
    ...
    20. HUM TGT: TYPE                  CIVILIAN: "JESUIT PRIESTS"  ← xref notation
    21. HUM TGT: NUMBER                6: "JESUIT PRIESTS"
    23. HUM TGT: EFFECT OF INCIDENT    DEATH: "JESUIT PRIESTS"

Xref notation: VALUE: "ENTITY" — only the VALUE part is kept.
OR alternatives: "ATTACK / BOMBING" — both values stored, eval.py handles them.

Usage:
    python build_gold_labels.py \\
        --key  /path/to/key-tst3.v2 \\
        --out  gold_labels_tst3.json

    # Or point at the raw data if it's a plain text key file:
    python build_gold_labels.py \\
        --key  /mnt/parscratch/users/acp25ck/team-rg1/data/raw/muc34/TEST/RESULTS/TST3/key-tst3.v2 \\
        --out  /mnt/parscratch/users/acp25ck/team-rg1/data/gold/gold_labels_tst3.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


# ── Slot ID → canonical slot name (for debug output only) ─────────────────────
SLOT_NAMES = {
    0: "MESSAGE ID", 1: "TEMPLATE ID",
    2: "INCIDENT: DATE", 3: "INCIDENT: LOCATION",
    4: "INCIDENT: TYPE", 5: "INCIDENT: STAGE OF EXECUTION",
    6: "INCIDENT: INSTRUMENT ID", 7: "INCIDENT: INSTRUMENT TYPE",
    8: "PERP: INCIDENT CATEGORY", 9: "PERP: INDIVIDUAL ID",
    10: "PERP: ORGANIZATION ID", 11: "PERP: ORGANIZATION CONFIDENCE",
    12: "PHYS TGT: ID", 13: "PHYS TGT: TYPE",
    14: "PHYS TGT: NUMBER", 15: "PHYS TGT: FOREIGN NATION",
    16: "PHYS TGT: EFFECT OF INCIDENT", 17: "PHYS TGT: TOTAL NUMBER",
    18: "HUM TGT: NAME", 19: "HUM TGT: DESCRIPTION",
    20: "HUM TGT: TYPE", 21: "HUM TGT: NUMBER",
    22: "HUM TGT: FOREIGN NATION", 23: "HUM TGT: EFFECT OF INCIDENT",
    24: "HUM TGT: TOTAL NUMBER",
}

# Regex: "NN.  SLOT NAME    VALUE"
# Handles both "4." and "4. " and captures everything after the slot name
_SLOT_LINE = re.compile(
    r"^\s*(\d+)\.\s+"   # slot number + dot
    r"[A-Z][A-Z0-9 :]+?"  # slot name (non-greedy)
    r"\s{2,}"            # two or more spaces separating name from value
    r"(.+?)\s*$",        # value (rest of line, stripped)
    re.IGNORECASE,
)

# Alternative simpler pattern: just "NN.  ... VALUE" where we split on the
# longest run of whitespace after the slot number
_SLOT_SIMPLE = re.compile(r"^\s*(\d+)\.\s+\S.*?\s{2,}(.+?)\s*$")

# Comment / event header line
_EVENT_HEADER = re.compile(r"^;\s*event\s+", re.IGNORECASE)

# Blank or comment-only line
_BLANK = re.compile(r"^\s*(;.*)?$")


def _strip_xref(value: str) -> str:
    """
    Remove xref notation from a slot value.

    MUC-3 xref format: VALUE: "ENTITY"  (possibly multiple entities)
    Examples:
        CIVILIAN: "JESUIT PRIESTS"          →  CIVILIAN
        DEATH: "JESUIT PRIESTS"             →  DEATH
        6: "SIX JESUITS"                    →  6
        SOME DAMAGE: "US EMBASSY"           →  SOME DAMAGE
        "CAR BOMB": "INSTRUMENT"            →  "CAR BOMB"   (string slots keep quotes)

    Rule: if the value contains `: "...` (colon followed by quoted entity),
    strip everything from the colon onwards.

    For OR alternatives like  CIVILIAN: "X" / MILITARY: "Y"
    this is applied per-branch BEFORE splitting on /.
    """
    # Strip each OR branch separately
    branches = re.split(r"\s*/\s*", value)
    cleaned = []
    for branch in branches:
        branch = branch.strip()
        # Remove xref: everything from ': "' onwards
        # Match: optional_value COLON SPACE* QUOTE ...
        m = re.match(r'^(.+?)\s*:\s*".*$', branch)
        if m:
            branch = m.group(1).strip()
        cleaned.append(branch)
    return " / ".join(cleaned)


def _parse_value(raw: str) -> str:
    """
    Clean a raw slot value:
    1. Strip xref notation
    2. Normalise whitespace
    3. Return "-" for null values
    """
    raw = raw.strip()
    if raw in {"-", "?", ""}:
        return "-"
    cleaned = _strip_xref(raw)
    if not cleaned or cleaned in {"-", "?"}:
        return "-"
    return cleaned


def _parse_slot_line(line: str):
    """
    Parse one slot line. Returns (slot_id: int, value: str) or None.
    """
    # Try primary pattern
    m = _SLOT_LINE.match(line)
    if not m:
        m = _SLOT_SIMPLE.match(line)
    if not m:
        return None
    slot_id = int(m.group(1))
    value   = _parse_value(m.group(2))
    return slot_id, value


def parse_key_file(path: str) -> dict:
    """
    Parse a MUC-3 key/response template file.

    Returns: {doc_id: [template_dict, ...]}
    where each template_dict has string keys "0"-"24".
    """
    result        = defaultdict(list)
    current_doc   = None
    current_tmpl  = None
    n_templates   = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    for lineno, line in enumerate(lines, 1):
        line = line.rstrip("\n")

        # Skip blank / comment lines (but not event headers)
        if _EVENT_HEADER.match(line):
            # New template starts
            # Flush previous template if any
            if current_doc is not None and current_tmpl is not None:
                # Fill missing slots with "-"
                for sid in range(25):
                    current_tmpl.setdefault(str(sid), "-")
                result[current_doc].append(current_tmpl)
                n_templates += 1
            current_tmpl = {}
            current_doc  = None  # will be set by slot 0
            continue

        if _BLANK.match(line):
            continue

        parsed = _parse_slot_line(line)
        if parsed is None:
            # Could not parse — skip silently (some files have continuation lines)
            continue

        slot_id, value = parsed

        if current_tmpl is None:
            # Shouldn't happen but be safe
            current_tmpl = {}

        current_tmpl[str(slot_id)] = value

        if slot_id == 0 and value != "-":
            current_doc = value.strip()

    # Flush final template
    if current_doc is not None and current_tmpl is not None:
        for sid in range(25):
            current_tmpl.setdefault(str(sid), "-")
        result[current_doc].append(current_tmpl)
        n_templates += 1

    print(f"  Parsed {len(result)} documents, {n_templates} templates")
    return dict(result)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Convert a MUC-3/4 key template file to gold_labels.json for eval.py. "
            "Handles xref notation (VALUE: \"ENTITY\") and OR alternatives."
        )
    )
    ap.add_argument("--key", required=True, metavar="PATH",
                    help="Path to the MUC-3 key template file (key-tst3.v2 or .ans)")
    ap.add_argument("--out", required=True, metavar="PATH",
                    help="Output path for gold_labels.json")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="Print first N templates to stdout for inspection (default: 0)")
    args = ap.parse_args()

    key_path = Path(args.key)
    if not key_path.exists():
        sys.exit(f"[ERROR] Key file not found: {args.key}")

    print(f"\nParsing: {args.key}")
    gold = parse_key_file(args.key)

    if args.show > 0:
        for doc_id, templates in list(gold.items())[:args.show]:
            print(f"\n── {doc_id} ({len(templates)} template(s)) ──")
            for tmpl in templates:
                for sid in range(25):
                    v = tmpl.get(str(sid), "-")
                    if v != "-":
                        print(f"  {sid:>2}. {SLOT_NAMES.get(sid,'?'):<36} {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(gold, fh, indent=2, ensure_ascii=False)
    print(f"  Written to: {args.out}")
    print("\nDone. Now run:")
    print(f"  python eval.py --pred predictions.json --gold {args.out} ...")


if __name__ == "__main__":
    main()
