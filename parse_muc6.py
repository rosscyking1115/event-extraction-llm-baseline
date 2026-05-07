#!/usr/bin/env python3
"""
MUC-6 Corporate Succession (ST task) data parser.

Parses per-document ST.key_XXXX files (nested MUC object format) and
corresponding WSJ article texts, outputs flat structured JSON ready for
LLM prompting and evaluation.

Usage:
    python parse_muc6.py --data_dir /path/to/muc_6 --output_dir /path/to/output

Input structure:
    muc_6/data/all-texts/wsj_text_0001 ... wsj_text_0100  (article texts)
    muc_6/data/evaluation/ST.key_0001  ... ST.key_0100    (gold succession templates)

ST key format (nested object graph):
    <TEMPLATE-docid-N> :=
        DOC_NR: "docid"
        CONTENT: <SUCCESSION_EVENT-docid-N>
    <SUCCESSION_EVENT-docid-N> :=
        SUCCESSION_ORG: <ORGANIZATION-docid-N>
        POST: "job title"
        IN_AND_OUT: <IN_AND_OUT-docid-N>
        VACANCY_REASON: RETIREMENT | FIRED | RESIGNED | REASSIGNMENT | NEW_POSITION | UNKNOWN
    <IN_AND_OUT-docid-N> :=
        IO_PERSON: <PERSON-docid-N>
        NEW_STATUS: IN | OUT
        ON_THE_JOB: YES | NO | UNCLEAR
        OTHER_ORG: <ORGANIZATION-docid-N>    (optional)
        REL_OTHER_ORG: SAME_ORG | RELATED_ORG | OTHER_ORG  (optional)
    <PERSON-docid-N> :=
        PER_NAME: "Full Name"
        PER_ALIAS: "Alias"
        PER_TITLE: "Mr." | "Ms." | ...
    <ORGANIZATION-docid-N> :=
        ORG_NAME: "Org Name"
        ORG_TYPE: COMPANY | GOVERNMENT | ...

Output flat schema per succession event:
    succession_org    : str | null   (ORG_NAME of SUCCESSION_ORG)
    post              : str | null   (job title)
    vacancy_reason    : str | null
    person_in         : str | null   (PER_NAME of IN person, or list if multiple)
    person_out        : str | null   (PER_NAME of OUT person, or list if multiple)
    on_the_job_in     : str | null   (YES/NO/UNCLEAR for IN person)
    on_the_job_out    : str | null   (YES/NO/UNCLEAR for OUT person)
    other_org_in      : str | null   (ORG_NAME of OTHER_ORG for IN person)
    rel_other_org_in  : str | null
"""

import os
import re
import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level object-graph parser
# ---------------------------------------------------------------------------

def parse_object_graph(content):
    """
    Parse the full MUC object-graph notation into a dict of dicts.

    Returns:
        objects: {object_id: {field_name: [value1, value2, ...]}}

    Values are raw strings; references look like "<TYPE-docid-N>".
    Multiple alternatives (separated by " / " or on lines starting with "/")
    are stored as separate list entries.
    """
    objects = {}

    # Split on object headers: "<ID> :="  followed by a newline
    obj_header_re = re.compile(r'<([^>]+)>\s*:=\s*\n')
    header_positions = [(m.group(1).strip(), m.end())
                        for m in obj_header_re.finditer(content)]

    for i, (obj_id, body_start) in enumerate(header_positions):
        body_end = header_positions[i + 1][1] - len(header_positions[i + 1][0]) - 5 \
            if i + 1 < len(header_positions) else len(content)
        body = content[body_start:body_end]

        fields = {}
        current_field = None
        current_values = []

        for line in body.split('\n'):
            raw = line.rstrip()
            stripped = raw.strip()

            if not stripped:
                continue

            # Field line: starts with 4 spaces then FIELD_NAME:
            field_match = re.match(r'    (\w+(?:_\w+)*):\s*(.*)', raw)
            if field_match:
                if current_field is not None:
                    fields[current_field] = current_values
                current_field = field_match.group(1)
                val = field_match.group(2).strip()
                current_values = _split_alternatives(val) if val else []
            elif current_field is not None and stripped:
                # Continuation / alternate value
                # Lines starting with "/" are alternatives
                alt = stripped.lstrip('/')
                alt_vals = _split_alternatives(alt.strip())
                current_values.extend(alt_vals)

        if current_field is not None:
            fields[current_field] = current_values

        objects[obj_id] = fields

    return objects


def _split_alternatives(val_str):
    """
    Split a value string on ' / ' to get alternative values.
    Filters out empty strings.
    """
    parts = [p.strip() for p in val_str.split(' / ')]
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Reference resolution helpers
# ---------------------------------------------------------------------------

def resolve_ref(ref_str, objects):
    """
    Given a string like '<ORGANIZATION-9301060123-2>', return the object dict.
    Returns {} if not found.
    """
    m = re.match(r'<([^>]+)>', ref_str.strip())
    if m:
        return objects.get(m.group(1).strip(), {})
    return {}


def first_value(field_values, strip_quotes=True):
    """
    Return the first non-empty value from a field's value list.
    Optionally strips surrounding double-quotes.
    """
    for v in (field_values or []):
        v = v.strip()
        if v:
            if strip_quotes:
                v = v.strip('"')
            return v or None
    return None


def all_values(field_values, strip_quotes=True):
    """Return all non-empty values from a field's value list."""
    result = []
    for v in (field_values or []):
        v = v.strip()
        if v:
            if strip_quotes:
                v = v.strip('"')
            if v:
                result.append(v)
    return result


# ---------------------------------------------------------------------------
# Succession event flattening
# ---------------------------------------------------------------------------

def flatten_in_and_out(in_out_ref, objects):
    """
    Resolve an IN_AND_OUT reference and return a simple dict with:
        new_status, on_the_job, person_name, other_org_name, rel_other_org
    """
    io_obj = resolve_ref(in_out_ref, objects)
    if not io_obj:
        return None

    new_status = first_value(io_obj.get('NEW_STATUS', []), strip_quotes=False)
    on_the_job = first_value(io_obj.get('ON_THE_JOB', []), strip_quotes=False)

    person_name = None
    person_refs = io_obj.get('IO_PERSON', [])
    if person_refs:
        person_obj = resolve_ref(person_refs[0], objects)
        person_name = first_value(person_obj.get('PER_NAME', []))

    other_org_name = None
    other_org_refs = io_obj.get('OTHER_ORG', [])
    if other_org_refs:
        other_org_obj = resolve_ref(other_org_refs[0], objects)
        other_org_name = first_value(other_org_obj.get('ORG_NAME', []))

    rel_other_org = first_value(io_obj.get('REL_OTHER_ORG', []), strip_quotes=False)

    return {
        'new_status': new_status,
        'on_the_job': on_the_job,
        'person_name': person_name,
        'other_org_name': other_org_name,
        'rel_other_org': rel_other_org,
    }


def flatten_succession_event(event_ref_or_id, objects):
    """
    Resolve a SUCCESSION_EVENT reference and flatten to a simple dict.

    Returns a dict with keys matching the output schema.
    """
    # Accept either a reference string "<X>" or a bare object ID
    if event_ref_or_id.startswith('<'):
        event_obj = resolve_ref(event_ref_or_id, objects)
    else:
        event_obj = objects.get(event_ref_or_id, {})

    if not event_obj:
        return None

    # Succession org
    org_refs = event_obj.get('SUCCESSION_ORG', [])
    succession_org = None
    if org_refs:
        org_obj = resolve_ref(org_refs[0], objects)
        succession_org = first_value(org_obj.get('ORG_NAME', []))

    # Post (job title) — take first alternative
    post = first_value(event_obj.get('POST', []))

    # Vacancy reason
    vacancy_reason = first_value(event_obj.get('VACANCY_REASON', []),
                                 strip_quotes=False)

    # Resolve all IN_AND_OUT entries
    in_out_refs = event_obj.get('IN_AND_OUT', [])
    in_persons = []
    out_persons = []
    on_job_in_list = []
    on_job_out_list = []
    other_org_in = None
    rel_other_org_in = None

    for ref in in_out_refs:
        io = flatten_in_and_out(ref, objects)
        if io is None:
            continue
        if io['new_status'] == 'IN':
            if io['person_name']:
                in_persons.append(io['person_name'])
            on_job_in_list.append(io['on_the_job'])
            if io['other_org_name'] and other_org_in is None:
                other_org_in = io['other_org_name']
            if io['rel_other_org'] and rel_other_org_in is None:
                rel_other_org_in = io['rel_other_org']
        elif io['new_status'] == 'OUT':
            if io['person_name']:
                out_persons.append(io['person_name'])
            on_job_out_list.append(io['on_the_job'])

    def _one_or_list(lst):
        if not lst:
            return None
        return lst[0] if len(lst) == 1 else lst

    return {
        'succession_org': succession_org,
        'post': post,
        'vacancy_reason': vacancy_reason,
        'person_in': _one_or_list(in_persons),
        'person_out': _one_or_list(out_persons),
        'on_the_job_in': _one_or_list(on_job_in_list),
        'on_the_job_out': _one_or_list(on_job_out_list),
        'other_org_in': other_org_in,
        'rel_other_org_in': rel_other_org_in,
    }


# ---------------------------------------------------------------------------
# Top-level file parser
# ---------------------------------------------------------------------------

def parse_st_key_file(key_file_path):
    """
    Parse a single MUC-6 ST.key file.

    Returns:
        doc_nr              : str (document number from DOC_NR field)
        succession_events   : list of flattened event dicts
    """
    with open(key_file_path, 'r', encoding='latin-1') as f:
        content = f.read()

    objects = parse_object_graph(content)

    doc_nr = None
    succession_events = []

    # Find the TEMPLATE object — it's the root
    for obj_id, fields in objects.items():
        if obj_id.startswith('TEMPLATE-'):
            # Extract doc number
            doc_nr_vals = fields.get('DOC_NR', [])
            if doc_nr_vals and doc_nr is None:
                doc_nr = doc_nr_vals[0].strip('"').strip()

            # Extract succession event references from CONTENT field
            content_refs = fields.get('CONTENT', [])
            for ref in content_refs:
                ref = ref.strip()
                if ref.startswith('<') and 'SUCCESSION_EVENT' in ref:
                    event = flatten_succession_event(ref, objects)
                    if event is not None:
                        succession_events.append(event)

    return doc_nr, succession_events


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir):
    """
    Build the full MUC-6 ST dataset by combining all key files with texts.

    Returns list of dicts:
        doc_id              : str
        file_num            : str  (zero-padded 4-digit number from filename)
        split               : "test"
        text                : str  (WSJ article text)
        succession_events   : list of flattened event dicts
        has_event           : bool
        n_events            : int
    """
    data_dir = Path(data_dir)
    eval_dir = data_dir / "data" / "evaluation"
    text_dir = data_dir / "data" / "all-texts"

    st_key_files = sorted(eval_dir.glob("ST.key_*"))
    print(f"  Found {len(st_key_files)} ST key files")

    dataset = []

    for key_file in st_key_files:
        file_num = key_file.name.split('_')[1]  # e.g. "0001"

        doc_nr, succession_events = parse_st_key_file(key_file)

        # Load article text
        text_file = text_dir / f"wsj_text_{file_num}"
        text = ""
        if text_file.exists():
            with open(text_file, 'r', encoding='latin-1') as f:
                text = f.read().strip()

        doc_id = doc_nr if doc_nr else f"wsj_{file_num}"

        entry = {
            "doc_id": doc_id,
            "file_num": file_num,
            "split": "test",
            "text": text,
            "succession_events": succession_events,
            "has_event": len(succession_events) > 0,
            "n_events": len(succession_events),
        }
        dataset.append(entry)

    return dataset


def print_stats(dataset):
    """Print summary statistics for the parsed dataset."""
    n_docs = len(dataset)
    n_with_event = sum(1 for d in dataset if d['has_event'])
    total_events = sum(d['n_events'] for d in dataset)

    print(f"\n  === MUC-6 ST Stats ===")
    print(f"  Documents           : {n_docs}")
    print(f"  With events         : {n_with_event} ({100*n_with_event/n_docs:.0f}%)")
    print(f"  Total events        : {total_events}")
    print(f"  Avg events/doc      : {total_events/n_docs:.2f}")

    # Slot fill rates
    slot_keys = ['succession_org', 'post', 'vacancy_reason',
                 'person_in', 'person_out', 'on_the_job_in', 'on_the_job_out',
                 'other_org_in', 'rel_other_org_in']
    slot_counts = {k: 0 for k in slot_keys}
    for doc in dataset:
        for ev in doc['succession_events']:
            for k in slot_keys:
                v = ev.get(k)
                if v is not None and v != []:
                    slot_counts[k] += 1

    print(f"\n  Slot fill rates (out of {total_events} events):")
    if total_events > 0:
        for slot_name, count in slot_counts.items():
            pct = 100 * count / total_events
            bar = '#' * int(pct / 5)
            print(f"    {slot_name:<25} {count:4d} ({pct:5.1f}%) {bar}")

    # Vacancy reason distribution
    from collections import Counter
    vr_counts = Counter()
    for doc in dataset:
        for ev in doc['succession_events']:
            vr = ev.get('vacancy_reason') or 'UNKNOWN'
            vr_counts[vr] += 1
    print(f"\n  Vacancy reason distribution:")
    for vr, cnt in vr_counts.most_common():
        print(f"    {vr:<25} {cnt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse MUC-6 ST (corporate succession) data into JSON"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to muc_6 dataset directory (contains data/all-texts and data/evaluation)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print detailed statistics after parsing"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing MUC-6 ST data...")
    dataset = build_dataset(args.data_dir)

    out_path = os.path.join(args.output_dir, "muc6_test.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    n_with_event = sum(1 for d in dataset if d['has_event'])
    total_events = sum(d['n_events'] for d in dataset)
    print(f"Saved {len(dataset)} documents -> {out_path}")
    print(f"  {n_with_event}/{len(dataset)} docs have events, {total_events} total events")

    if args.stats:
        print_stats(dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
