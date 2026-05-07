#!/usr/bin/env python3
"""
MUC-3/4 data parser.

Decompresses .tar.Z corpus files, parses gold key files and article texts,
outputs structured JSON ready for LLM prompting and evaluation.

Usage:
    python parse_muc34.py --data_dir /path/to/muc34 --output_dir /path/to/output
    python parse_muc34.py --data_dir /path/to/muc34 --output_dir /path/to/output --splits tst3 tst4

Gold key slot numbering (0-24):
    0  MESSAGE: ID                  -> MESSAGE_ID
    1  MESSAGE: TEMPLATE            -> MESSAGE_TEMPLATE (1/2/3... or * = no event)
    2  INCIDENT: DATE               -> INCIDENT_DATE
    3  INCIDENT: LOCATION           -> INCIDENT_LOCATION
    4  INCIDENT: TYPE               -> INCIDENT_TYPE
    5  INCIDENT: STAGE OF EXECUTION -> INCIDENT_STAGE
    6  INCIDENT: INSTRUMENT ID      -> INCIDENT_INSTRUMENT_ID
    7  INCIDENT: INSTRUMENT TYPE    -> INCIDENT_INSTRUMENT_TYPE
    8  PERP: INCIDENT CATEGORY      -> PERP_INCIDENT_CATEGORY
    9  PERP: INDIVIDUAL ID          -> PERP_INDIVIDUAL_ID
    10 PERP: ORGANIZATION ID        -> PERP_ORGANIZATION_ID
    11 PERP: ORGANIZATION CONFIDENCE-> PERP_ORG_CONFIDENCE
    12 PHYS TGT: ID                 -> PHYS_TGT_ID
    13 PHYS TGT: TYPE               -> PHYS_TGT_TYPE
    14 PHYS TGT: NUMBER             -> PHYS_TGT_NUMBER
    15 PHYS TGT: FOREIGN NATION     -> PHYS_TGT_FOREIGN_NATION
    16 PHYS TGT: EFFECT OF INCIDENT -> PHYS_TGT_EFFECT
    17 PHYS TGT: TOTAL NUMBER       -> PHYS_TGT_TOTAL_NUMBER
    18 HUM TGT: NAME                -> HUM_TGT_NAME
    19 HUM TGT: DESCRIPTION         -> HUM_TGT_DESCRIPTION
    20 HUM TGT: TYPE                -> HUM_TGT_TYPE
    21 HUM TGT: NUMBER              -> HUM_TGT_NUMBER
    22 HUM TGT: FOREIGN NATION      -> HUM_TGT_FOREIGN_NATION
    23 HUM TGT: EFFECT OF INCIDENT  -> HUM_TGT_EFFECT
    24 HUM TGT: TOTAL NUMBER        -> HUM_TGT_TOTAL_NUMBER

Slot value conventions:
    "-"  = slot is empty/null for this event
    "*"  = not applicable (used when MESSAGE_TEMPLATE is *)
    Values may be multi-line (continuation lines) for multi-valued slots.
    Values may be prefixed with a type: e.g. "CIVILIAN: \"JESUIT PRIESTS\""
    Multiple alternatives separated by " / " on the same or next lines.
"""

import os
import re
import json
import subprocess
import tempfile
import argparse
from pathlib import Path


# Canonical slot names keyed by slot number
MUC4_SLOT_NAMES = {
    0: "MESSAGE_ID",
    1: "MESSAGE_TEMPLATE",
    2: "INCIDENT_DATE",
    3: "INCIDENT_LOCATION",
    4: "INCIDENT_TYPE",
    5: "INCIDENT_STAGE",
    6: "INCIDENT_INSTRUMENT_ID",
    7: "INCIDENT_INSTRUMENT_TYPE",
    8: "PERP_INCIDENT_CATEGORY",
    9: "PERP_INDIVIDUAL_ID",
    10: "PERP_ORGANIZATION_ID",
    11: "PERP_ORG_CONFIDENCE",
    12: "PHYS_TGT_ID",
    13: "PHYS_TGT_TYPE",
    14: "PHYS_TGT_NUMBER",
    15: "PHYS_TGT_FOREIGN_NATION",
    16: "PHYS_TGT_EFFECT",
    17: "PHYS_TGT_TOTAL_NUMBER",
    18: "HUM_TGT_NAME",
    19: "HUM_TGT_DESCRIPTION",
    20: "HUM_TGT_TYPE",
    21: "HUM_TGT_NUMBER",
    22: "HUM_TGT_FOREIGN_NATION",
    23: "HUM_TGT_EFFECT",
    24: "HUM_TGT_TOTAL_NUMBER",
}

# Slots that are event-relevant (exclude metadata slots 0 and 1)
EVENT_SLOTS = [s for s in MUC4_SLOT_NAMES if s >= 2]

# Valid incident types in MUC-4
INCIDENT_TYPES = {
    "ATTACK", "BOMBING", "KIDNAPPING", "ARSON",
    "ASSASSINATION", "ROBBERY", "FORCED WORK STOPPAGE"
}


def decompress_tarZ(tar_z_path, output_dir):
    """Decompress a .tar.Z file using zcat | tar x"""
    result = subprocess.run(
        f'zcat "{tar_z_path}" | tar x',
        shell=True, cwd=output_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to decompress {tar_z_path}: {result.stderr}")


def parse_articles(corpus_file):
    """
    Parse all articles from a single MUC-4 corpus file (all docs concatenated).
    Returns dict: {doc_id: article_text}
    """
    docs = {}
    current_id = None
    current_lines = []

    # Doc ID pattern: TST3-MUC4-0001, TST4-MUC4-0001, DEV-MUC4-0001
    doc_id_pattern = re.compile(r'^(TST[1-4]|DEV)-MUC4-\d{4}$')

    with open(corpus_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.rstrip('\n')
            if doc_id_pattern.match(line.strip()):
                if current_id is not None:
                    docs[current_id] = '\n'.join(current_lines).strip()
                current_id = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

    if current_id is not None:
        docs[current_id] = '\n'.join(current_lines).strip()

    return docs


def normalise_slot_value(raw_value):
    """
    Normalise a raw slot value string.

    Returns None if the value is empty, "-", or "*".
    Returns a string with leading/trailing whitespace removed.
    Multi-line values (continuation lines) are joined with " | ".
    Quoted strings have their quotes preserved for now (normalisation
    happens at scoring time).
    """
    if raw_value is None:
        return None
    # Strip and check for null/inapplicable markers
    stripped = raw_value.strip()
    if stripped in ("-", "*", ""):
        return None
    # Join continuation lines
    parts = [p.strip() for p in stripped.split('\n') if p.strip()]
    parts = [p for p in parts if p not in ("-", "*")]
    if not parts:
        return None
    return " | ".join(parts)


def parse_gold_keys(key_file):
    """
    Parse a MUC-4 gold key file.

    Returns dict: {doc_id: [template_dict, ...]}
    Each template_dict maps slot names (from MUC4_SLOT_NAMES) to raw string values.
    A document with no relevant event has MESSAGE_TEMPLATE == "*".
    """
    with open(key_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    all_docs = {}   # doc_id -> list of template dicts
    current_block_lines = []
    blocks = []

    for line in lines:
        line = line.rstrip('\n')
        # A new template block starts when slot 0 appears
        if re.match(r'^0\.\s+MESSAGE: ID', line):
            if current_block_lines:
                blocks.append(current_block_lines)
            current_block_lines = [line]
        else:
            current_block_lines.append(line)

    if current_block_lines:
        blocks.append(current_block_lines)

    for block in blocks:
        template = {}
        current_slot_num = None
        current_value_lines = []

        for line in block:
            # Match a slot header line: "N.  SLOT: NAME   VALUE..."
            # Slot number followed by period, then 2+ spaces before value
            slot_match = re.match(r'^(\d+)\.\s{1,5}(.+?)\s{2,}(.*)', line)
            if slot_match:
                # Save previous slot
                if current_slot_num is not None:
                    slot_name = MUC4_SLOT_NAMES.get(current_slot_num,
                                                     f"SLOT_{current_slot_num}")
                    template[slot_name] = '\n'.join(current_value_lines)

                current_slot_num = int(slot_match.group(1))
                current_value_lines = [slot_match.group(3).strip()]

            elif current_slot_num is not None and line.strip():
                # Continuation / alternate value line (indented, not a slot header)
                if not re.match(r'^\d+\.', line):
                    current_value_lines.append(line.strip())

        # Save last slot
        if current_slot_num is not None:
            slot_name = MUC4_SLOT_NAMES.get(current_slot_num,
                                             f"SLOT_{current_slot_num}")
            template[slot_name] = '\n'.join(current_value_lines)

        if 'MESSAGE_ID' in template:
            doc_id = template['MESSAGE_ID'].strip()
            if doc_id not in all_docs:
                all_docs[doc_id] = []
            all_docs[doc_id].append(template)

    return all_docs


def template_to_record(template):
    """
    Convert a raw template dict to a clean record with normalised values.
    Returns a dict with all 25 slots, nulls where empty/inapplicable.
    """
    record = {}
    for slot_num, slot_name in MUC4_SLOT_NAMES.items():
        raw = template.get(slot_name, None)
        if slot_name in ("MESSAGE_ID", "MESSAGE_TEMPLATE"):
            # Keep these as-is (metadata)
            record[slot_name] = raw.strip() if raw else None
        else:
            record[slot_name] = normalise_slot_value(raw)
    return record


def build_dataset(corpus_file, key_file, split_name):
    """
    Combine article texts and gold key templates into a dataset.

    Returns list of dicts:
        doc_id       : str
        split        : str
        text         : str  (full article text)
        templates    : list of cleaned template records
        has_event    : bool (True if at least one template is not *)
        n_templates  : int
    """
    print(f"  Parsing articles from {corpus_file}...")
    articles = parse_articles(corpus_file)
    print(f"  Found {len(articles)} articles")

    print(f"  Parsing gold keys from {key_file}...")
    gold_keys = parse_gold_keys(key_file)
    print(f"  Found {len(gold_keys)} gold key entries")

    dataset = []
    for doc_id, text in sorted(articles.items()):
        raw_templates = gold_keys.get(doc_id, [])
        clean_templates = [template_to_record(t) for t in raw_templates]

        has_event = any(
            t.get('MESSAGE_TEMPLATE', '*') not in ('*', None)
            for t in clean_templates
        )

        entry = {
            "doc_id": doc_id,
            "split": split_name,
            "text": text,
            "templates": clean_templates,
            "has_event": has_event,
            "n_templates": len([t for t in clean_templates
                                 if t.get('MESSAGE_TEMPLATE', '*') not in ('*', None)])
        }
        dataset.append(entry)

    return dataset


def print_stats(dataset, split_name):
    """Print summary statistics for a parsed dataset."""
    n_docs = len(dataset)
    n_with_event = sum(1 for d in dataset if d['has_event'])
    n_templates = sum(d['n_templates'] for d in dataset)

    print(f"\n  === {split_name.upper()} Stats ===")
    print(f"  Documents      : {n_docs}")
    print(f"  With events    : {n_with_event} ({100*n_with_event/n_docs:.0f}%)")
    print(f"  Total templates: {n_templates}")

    # Count filled slots
    slot_counts = {name: 0 for name in MUC4_SLOT_NAMES.values() if name not in
                   ("MESSAGE_ID", "MESSAGE_TEMPLATE")}
    for doc in dataset:
        for tmpl in doc['templates']:
            for slot_name in slot_counts:
                if tmpl.get(slot_name) is not None:
                    slot_counts[slot_name] += 1

    print(f"\n  Slot fill rates (out of {n_templates} event templates):")
    if n_templates > 0:
        for slot_name, count in slot_counts.items():
            pct = 100 * count / n_templates
            bar = '#' * int(pct / 5)
            print(f"    {slot_name:<35} {count:4d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse MUC-3/4 data into structured JSON for LLM evaluation"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to muc34 dataset directory (contains TASK/ and TEST/)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["tst3", "tst4"],
        choices=["tst1", "tst2", "tst3", "tst4", "dev"],
        help="Which splits to parse (default: tst3 tst4)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print detailed slot fill statistics"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = Path(args.data_dir)
    corpora_dir = data_dir / "TASK" / "CORPORA"

    for split in args.splits:
        print(f"\nProcessing split: {split}")
        tar_z = corpora_dir / f"{split}.tar.Z"

        if not tar_z.exists():
            print(f"  WARNING: {tar_z} not found, skipping")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  Decompressing {tar_z.name}...")
            decompress_tarZ(str(tar_z), tmpdir)

            tmp_files = os.listdir(tmpdir)
            corpus_file = None
            key_file = None

            for fname in tmp_files:
                fpath = os.path.join(tmpdir, fname)
                if fname.startswith('key-'):
                    key_file = fpath
                elif not fname.endswith('.Z') and os.path.isfile(fpath):
                    corpus_file = fpath

            if not corpus_file:
                print(f"  ERROR: Could not find corpus file. Files: {tmp_files}")
                continue
            if not key_file:
                print(f"  ERROR: Could not find key file. Files: {tmp_files}")
                continue

            dataset = build_dataset(corpus_file, key_file, split)

            out_path = os.path.join(args.output_dir, f"muc4_{split}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            print(f"  Saved {len(dataset)} documents -> {out_path}")

            if args.stats:
                print_stats(dataset, split)

    print("\nDone.")


if __name__ == "__main__":
    main()
