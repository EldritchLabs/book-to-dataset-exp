# -*- coding: utf-8 -*-
"""Salvage rejected transcripts from rp_rejected.jsonl into ShareGPT dataset_final.jsonl format.

Usage:
    python rejection_salvager.py [--input work/rp_rejected.jsonl] [--output work/dataset_salvaged.jsonl]
"""
import argparse
import json
import os
import sys

DEFAULT_SYSTEM_TEMPLATE = (
    "You are {char} and the scene's narrator. Inhabit {char} completely, "
    "delivering authentic voice, world atmosphere, and sensory presence. "
    "Weave physical actions, reactions, and dialogue seamlessly. Drive the "
    "environment, pacing, and surroundings, but never act or speak for your "
    "interlocutor.\n\n"
    "<format>\n"
    "- Make each turn longer than 300 words.\n"
    "- Start the turn with the speaker's name and a colon.\n"
    "- Dialogue without quotation marks, woven into the prose.\n"
    "</format>."
)


def build_system_prompt(item):
    if item.get("system_prompt"):
        return item["system_prompt"]
    char = item.get("char") or "the character"
    return DEFAULT_SYSTEM_TEMPLATE.format(char=char)


def convert_item(item):
    turns = item.get("turns") or []
    conversations = [{"from": "system", "value": build_system_prompt(item)}]

    for turn in turns:
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            conversations.append({"from": "human", "value": text})
        elif role == "char":
            conversations.append({"from": "gpt", "value": text})

    # Ensure there is at least one non-system turn
    if len(conversations) < 2:
        return None

    return {"conversations": conversations}


def main():
    parser = argparse.ArgumentParser(description="Convert rp_rejected.jsonl to dataset_final.jsonl format.")
    parser.add_argument("--input", "-i", default="rp_rejected.jsonl", help="Path to rp_rejected.jsonl")
    parser.add_argument("--output", "-o", default="dataset_salvaged.jsonl", help="Path to output JSONL")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping line {line_no}: invalid JSON ({exc})", file=sys.stderr)
                skipped += 1
                continue

            converted = convert_item(item)
            if converted is None:
                skipped += 1
                continue

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done: {written}/{total} salvaged -> {args.output} (skipped {skipped})")


if __name__ == "__main__":
    main()