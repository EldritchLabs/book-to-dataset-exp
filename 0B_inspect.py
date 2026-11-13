# -*- coding: utf-8 -*-
"""Step 00 — inspect the source text before spending any GPU time on it.

Reports the structure of the file: how many blocks it has, how long they are,
how much direct speech there is, whether chapters and scene breaks are marked
up. The segmentation thresholds in the config are tuned against this report.

This step makes no LLM calls and takes a second. Run it first, always.

    python 00_inspect.py [--config config.yaml]
"""
import re

from pipeline.config import load_config, parse_args, resolve_input_path
from pipeline.files import read_text
from pipeline.lang import chapter_regex, load_pack
from pipeline.textstats import percentile

# A line made of repeated separator characters: * * *, ---, ~~~ and friends.
SCENE_BREAK_RE = re.compile(r"^\s*([*#~—\-–_=]\s*){3,}\s*$")
# A line that opens with a dialogue marker, in any of the common typographies,
# OR a colon-break lead-in line immediately followed by quoted dialogue.
DIALOGUE_START = re.compile(r'^\s*[—–\-«"„\u201c]')
COLON_LEAD_IN = re.compile(r':\s*$')
# Matches markdown headings (# through ####) and bold section titles (**Title**)
def build_header_regex(cfg):
    """Build a dynamic header regex from cfg['segment']['chapter_headers']."""
    headers = cfg.get("segment", {}).get("chapter_headers", ["#", "##", "###", "####"])
    if isinstance(headers, str):
        headers = [h.strip() for h in headers.split(",") if h.strip()]
    
    # Escape and sort by length descending to match longest prefixes first
    esc_headers = [re.escape(h) for h in headers if h]
    if esc_headers:
        pattern = r"^(?:(?:" + "|".join(esc_headers) + r")\s+(.+)$|\*\*[A-Z0-9].+\*\*\s*$)"
    else:
        pattern = r"^(?:#{1,4}\s+(.+)$|\*\*[A-Z0-9].+\*\*\s*$)"
    return re.compile(pattern, re.M)


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    # The heading words come from the language pack, so this diagnostic sees
    # exactly what step 01 will see.
    chapter_re = chapter_regex(pack)
    md_heading_re = build_header_regex(cfg)

    source_file = resolve_input_path(cfg)
    text = read_text(source_file, cfg["input"]["encodings"])
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    assert blocks, "the source text is empty after splitting on blank lines"
    lengths = sorted(len(b) for b in blocks)
    lines = text.split("\n")

    def has_dialogue(block):
        if DIALOGUE_START.match(block):
            return True
        b_lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        for i, ln in enumerate(b_lines):
            if DIALOGUE_START.match(ln):
                return True
            if COLON_LEAD_IN.search(ln) and i + 1 < len(b_lines) and DIALOGUE_START.match(b_lines[i + 1]):
                return True
        return False

    dialogue_blocks = [b for b in blocks if has_dialogue(b)]
    pure_dialogue = [
        b for b in blocks
        if len(b) > 400
        and sum(1 for ln in b.split("\n") if DIALOGUE_START.match(ln.strip())) / max(1, len(b.split("\n"))) > 0.8
    ]
    chapters = [
        ln.strip() for ln in lines 
        if (chapter_re.match(ln) or md_heading_re.match(ln.strip())) and len(ln.strip()) < 120
    ]
    breaks = [ln for ln in lines if SCENE_BREAK_RE.match(ln)]

    print(f"File: {source_file}, {len(text)} characters")
    print(f"Blocks separated by blank lines: {len(blocks)}")
    print(f"Block length in characters: min {lengths[0]} | p25 {percentile(lengths, .25)} | "
          f"median {percentile(lengths, .5)} | p75 {percentile(lengths, .75)} | "
          f"p95 {percentile(lengths, .95)} | max {lengths[-1]}")
    print(f"Blocks over 5000 characters: {sum(1 for x in lengths if x > 5000)}")
    print(f"Blocks opening with direct speech: {len(dialogue_blocks)} "
          f"({100 * len(dialogue_blocks) // max(1, len(blocks))}%)")
    print(f"Large dialogue-only blocks: {len(pure_dialogue)}")
    print(f"Chapter headings found: {len(chapters)}"
          + (f", for example: {chapters[:3]}" if chapters else ""))
    print(f"Explicit scene breaks found: {len(breaks)}")

    segment = cfg["segment"]
    print("\n--- what this means for the config ---")
    if percentile(lengths, .5) < 200:
        print("* The median block is short, so these are ordinary paragraphs. "
              "The LLM segmentation pass will do most of the work.")
    if lengths[-1] > segment["max_block_chars"]:
        over = sum(1 for x in lengths if x > segment["max_block_chars"])
        print(f"* {over} blocks are longer than segment.max_block_chars "
              f"({segment['max_block_chars']}), so they go through the LLM. "
              f"That is the part of step 01 that costs time.")
    if not breaks and not chapters:
        print("* Neither chapters nor scene breaks are marked up: the whole file "
              "goes through the LLM pass. Expect step 01 to be slow.")
    if breaks or chapters:
        print(f"* Structural markers are present ({len(chapters)} headings, "
              f"{len(breaks)} breaks), so much of the segmentation is free.")
    if len(dialogue_blocks) < len(blocks) // 20:
        print("* There is very little direct speech. This pipeline builds dialogue "
              "transcripts, so a source this narrative-heavy will yield few rows. "
              "Consider lowering convert.min_dialogue_share, and expect step 03 "
              "to skip a lot of scenes.")


if __name__ == "__main__":
    main()
