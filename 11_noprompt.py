# -*- coding: utf-8 -*-
"""Step 11 — Strip system prompts for weight-hardcoded SFT.

Removes all `system` messages from ShareGPT (or OpenAI `messages`) JSONL
datasets and writes the stripped files into `noprompt/`. Training without
a system prompt bakes the persona, tone, and style directly into the
model's weights.

    python 11_noprompt.py [--config config.yaml]
    python 11_noprompt.py --input dataset_final.jsonl
"""
import os
import sys

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import read_jsonl, write_jsonl


def strip_system_prompts(rows):
    """Remove system turns from each session while keeping valid conversations."""
    out = []
    stripped_msgs = 0
    for row in rows:
        if "conversations" in row and isinstance(row["conversations"], list):
            kept = [m for m in row["conversations"] if m.get("from") != "system"]
            stripped_msgs += len(row["conversations"]) - len(kept)
            if any(m.get("from") == "gpt" for m in kept):
                out.append({**row, "conversations": kept})
        elif "messages" in row and isinstance(row["messages"], list):
            kept = [m for m in row["messages"] if m.get("role") != "system"]
            stripped_msgs += len(row["messages"]) - len(kept)
            if any(m.get("role") == "assistant" for m in kept):
                out.append({**row, "messages": kept})
    return out, stripped_msgs


def main():
    def extra(parser):
        parser.add_argument(
            "--input",
            default=None,
            help="JSONL file to strip (default: antislop output if enabled, else assemble.output)",
        )
        parser.add_argument(
            "--out-dir",
            default="noprompt",
            help="Output directory (default: noprompt)",
        )

    args = parse_args(__doc__.splitlines()[0], extra)
    cfg = load_config(args.config)

    if args.input:
        source = args.input
    else:
        anti_cfg = cfg.get("antislop", {})
        anti_path = work_path(cfg, anti_cfg.get("output", "dataset_antislop.jsonl"))
        if anti_cfg.get("enabled") and os.path.exists(anti_path):
            source = anti_path
        else:
            source = cfg["assemble"]["output"]

    if not os.path.exists(source):
        sys.exit(f"\n{source} not found. Run 08_assemble.py (or pass --input PATH) first.\n")

    rows = read_jsonl(source)
    assert rows, f"{source} is empty"

    cleaned, stripped_msgs = strip_system_prompts(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, os.path.basename(source))
    write_jsonl(out_path, cleaned)

    print(
        f"Done: {len(cleaned)}/{len(rows)} sessions kept, "
        f"{stripped_msgs} system prompt(s) removed -> {out_path}"
    )


if __name__ == "__main__":
    main()