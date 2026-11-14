# -*- coding: utf-8 -*-
"""Step 01 — cut the source text into scenes.

Two passes, in this order:

  1. A structural pass, which is free: chapter headings, explicit scene breaks,
     blank-line blocks. Most well-formatted books are almost fully segmented by
     this alone.
  2. An LLM pass, which is not free, and therefore runs only inside blocks that
     have no structural markers and are longer than segment.max_block_chars.
     The model receives numbered paragraphs and returns the numbers of the
     paragraphs before which a new scene starts.

The result is post-processed: scenes below the minimum length are glued to
their neighbour, scenes above the maximum are split on paragraph borders.

    Output: <workdir>/scenes.jsonl   {id, chapter, text, chars}

    python 01_segment.py [--config config.yaml]
"""
import json
import os
import re
import sys

from pipeline.config import load_config, parse_args, resolve_input_path, work_path
from pipeline.files import append_jsonl, read_jsonl, read_text, write_jsonl
from pipeline.lang import chapter_regex, load_pack, render
from pipeline.llm import LLM, extract_json

SCENE_BREAK_RE = re.compile(r"^\s*([*#~—\-–_=]\s*){3,}\s*$")

# Keeps the terminator with the sentence it ends. The optional whitespace matters:
# CJK punctuation is not followed by a space.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s*")


def normalize(text):
    """Unify line endings and collapse runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


MD_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$")
SECTION_HEADING_RE = re.compile(r"^\*\*(?:\"[^\"]+\"|[^*]+)\*\*\s*$")

def build_header_regex(cfg):
    """Build a dynamic header regex from cfg['segment']['chapter_headers']."""
    headers = cfg.get("segment", {}).get("chapter_headers", ["#", "##", "###", "####"])
    if isinstance(headers, str):
        headers = [h.strip() for h in headers.split(",") if h.strip()]
    esc_headers = [re.escape(h) for h in headers if h]
    if esc_headers:
        pattern = r"^(?:" + "|".join(esc_headers) + r")\s+(.+)$"
    else:
        pattern = r"^#{1,4}\s+(.+)$"
    return re.compile(pattern)

def structural_blocks(text, chapter_re, md_heading_re):
    """Split on chapter headings, Markdown headers, bold section titles, and explicit scene breaks."""
    blocks, current, chapter = [], [], ""
    for line in text.split("\n"):
        stripped = line.strip()
        md_match = md_heading_re.match(stripped)
        sec_match = SECTION_HEADING_RE.match(stripped)
        
        is_md_header = bool(md_match and len(stripped) < 150)
        is_sec_header = bool(sec_match and len(stripped) < 100 and not stripped.startswith("**Intro"))
        is_txt_chapter = bool(chapter_re.match(line) and len(stripped) < 100)

        if is_md_header or is_sec_header or is_txt_chapter:
            if current and "".join(current).strip():
                blocks.append((chapter, "\n".join(current).strip()))
            heading_title = md_match.group(1).strip() if md_match else stripped.strip("*").strip('"').strip()
            current, chapter = [], heading_title
        elif SCENE_BREAK_RE.match(line):
            if current and "".join(current).strip():
                blocks.append((chapter, "\n".join(current).strip()))
            current = []
        else:
            current.append(line)
    if current and "".join(current).strip():
        blocks.append((chapter, "\n".join(current).strip()))
    return blocks


def paragraphs_of(block):
    return [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]


def llm_boundaries(llm, pack, paragraphs, window_chars, overlap, debug_dir, block_id=0):
    """Find scene boundaries in one long block, checkpointing progress to disk per window."""
    os.makedirs(debug_dir, exist_ok=True)
    state_file = os.path.join(debug_dir, f"seg_checkpoint_block_{block_id}.json")
    
    boundaries = set()
    start = 0
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                boundaries = set(saved.get("boundaries", []))
                start = saved.get("last_start", 0)
                print(f"  [resume] Block {block_id}: loaded {len(boundaries)} boundaries, resuming at paragraph {start}")
        except Exception:
            pass

    total_p = len(paragraphs)
    while start < total_p:
        end, budget = start, 0
        while end < total_p and (budget + len(paragraphs[end]) <= window_chars or end - start < 2):
            budget += len(paragraphs[end]) + 8
            end += 1

        pct = (start / total_p) * 100 if total_p else 0
        print(f"  [segmenting] block {block_id}: paragraphs {start}-{end} of {total_p} ({pct:.1f}%) ...", flush=True)
        numbered = "\n\n".join(f"[{i}] {paragraphs[i]}" for i in range(start, end))
        data, raw = None, ""
        for attempt in range(2):
            suffix = pack.require("segment.strict_suffix") if attempt else ""
            raw = llm.chat(
                pack.require("segment.system"),
                render(pack.require("segment.user"), paragraphs=numbered) + suffix,
                temperature=0.0,
                max_tokens=4096 if attempt == 0 else 8192,
            )
            try:
                data = extract_json(raw)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt:
                    debug_file = os.path.join(debug_dir, f"segfail_{start}_{end}.txt")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(raw)
                    print(f"  [warn] window {start}-{end}: unparseable after retry ({exc}); "
                          f"raw reply saved to {debug_file}", file=sys.stderr)

        if data is not None:
            raw_boundaries = []
            if isinstance(data, list):
                raw_boundaries = data
            elif isinstance(data, dict):
                raw_boundaries = data.get("boundaries", [])

            for value in raw_boundaries:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if start < index < end:
                    boundaries.add(index)

        # Checkpoint to JSON after every successful window with atomic write & flush
        next_start = end if end == total_p else max(start + 1, end - overlap)
        
        checkpoint_data = {
            "block_id": block_id,
            "last_start": next_start,
            "total_paragraphs": total_p,
            "percent_complete": round((next_start / total_p) * 100, 2),
            "boundaries_found": len(boundaries),
            "boundaries": sorted(list(boundaries))
        }
        
        tmp_file = state_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, state_file)

        # Also write a human-readable progress JSON containing the actual text of scenes discovered so far
        progress_scenes_file = os.path.join(debug_dir, f"scenes_discovered_block_{block_id}.json")
        current_scenes = split_by_boundaries(paragraphs[:end], boundaries)
        tmp_scenes = progress_scenes_file + ".tmp"
        with open(tmp_scenes, "w", encoding="utf-8") as f:
            json.dump({
                "block_id": block_id,
                "scenes_count": len(current_scenes),
                "scenes": [{"scene_no": i, "chars": len(sc), "preview": sc[:200].replace("\n", " ") + "..."} for i, sc in enumerate(current_scenes)]
            }, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_scenes, progress_scenes_file)

        print(f"    [saved checkpoint] {len(boundaries)} boundaries -> {state_file}", flush=True)

        if end == total_p:
            break
        start = next_start

    return boundaries


def split_by_boundaries(paragraphs, boundaries):
    scenes, current = [], []
    for index, paragraph in enumerate(paragraphs):
        if index in boundaries and current:
            scenes.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        scenes.append("\n\n".join(current))
    return scenes


def merge_short(scenes, min_chars):
    """Glue undersized scenes to their neighbour while preserving chapter metadata."""
    out = []
    for item in scenes:
        # Support both (chapter, text) tuples and bare text strings
        chap, txt = item if isinstance(item, (tuple, list)) else ("", item)
        if out:
            prev_chap, prev_txt = out[-1]
            # Only merge if either is under min_chars AND they belong to the same chapter
            # (or if either chapter is unspecified)
            same_chap = (prev_chap == chap or not prev_chap or not chap)
            if (len(txt) < min_chars or len(prev_txt) < min_chars) and same_chap:
                out[-1] = (prev_chap or chap, prev_txt + "\n\n" + txt)
                continue
        out.append((chap, txt))
    return out


def hard_split(text, max_chars):
    """Last-resort split of a single oversized paragraph.

    Paragraph borders are the preferred cut, but some files have none: a text
    dumped without blank lines is one paragraph from the first character to the
    last. Without this fallback such a file produces a single scene the size of
    the whole book, which then overflows the context window in step 02 and fails
    there instead of here, with a much less obvious error.

    Sentence borders are tried first. A "sentence" longer than the cap on its own
    is split on whitespace, and text with no whitespace at all is cut at the cap.
    """
    pieces, current = [], ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
            continue
        if current:
            pieces.append(current.strip())
            current = ""
        while len(sentence) > max_chars:
            # Prefer a space near the cap so a word is not cut in half.
            cut = sentence.rfind(" ", int(max_chars * 0.8), max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].lstrip()
        current = sentence
    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def cap_long(scenes, max_chars):
    """Split oversized scenes on paragraph borders while preserving chapter metadata.
    
    Never inside a paragraph while a paragraph border is available: a scene that
    ends mid-sentence produces a transcript that ends mid-sentence. When a single
    paragraph is itself over the cap, hard_split takes over."""
    out = []
    for item in scenes:
        chap, txt = item if isinstance(item, (tuple, list)) else ("", item)
        if len(txt) <= max_chars:
            out.append((chap, txt))
            continue
        current, current_len = [], 0
        for paragraph in paragraphs_of(txt):
            for piece in ([paragraph] if len(paragraph) <= max_chars
                          else hard_split(paragraph, max_chars)):
                if current_len + len(piece) > max_chars and current:
                    out.append((chap, "\n\n".join(current)))
                    current, current_len = [], 0
                current.append(piece)
                current_len += len(piece) + 2
        if current:
            out.append((chap, "\n\n".join(current)))
    return out


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    segment = cfg["segment"]
    llm = LLM(cfg, step_name="segment")

    source_file = resolve_input_path(cfg)
    print(f"Segmenting source: {source_file}")
    text = normalize(read_text(source_file, cfg["input"]["encodings"]))

    # v5.1: {{[INPUT]}} / {{[OUTPUT]}} pair mode
    io_matches = re.findall(
        r"\{\{\[INPUT\]\}\}(.*?)\{\{\[OUTPUT\]\}\}(.*?)(?=\{\{\[INPUT\]\}\}|\Z)",
        text,
        re.S,
    )
    if io_matches:
        output = work_path(cfg, "scenes.jsonl")
        existing = read_jsonl(output)
        done = {item["id"] for item in existing}
        if done:
            print(f"[resume] {len(done)}/{len(io_matches)} pairs already segmented, continuing")
        else:
            print(f"Detected {len(io_matches)} {{{{[INPUT]}}}}/{{{{[OUTPUT]}}}} pairs — synthesizing custom system prompts...")

        debug_dir = work_path(cfg, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        n_new = 0
        for scene_id, (raw_in, raw_out) in enumerate(io_matches):
            if scene_id in done:
                continue
            inp = raw_in.strip()
            out = raw_out.strip()
            if not out:
                continue
            sys_prompt = ""
            try:
                sys_prompt = llm.chat(
                    "You are an expert prompt engineer. Given a user story-seed/input and the "
                    "assistant's opening/output, write a concise, atmospheric system prompt (2-4 sentences) "
                    "that instructs a roleplay/storytelling model on the tone, setting, and persona.",
                    f"INPUT:\n{inp[:1500]}\n\nOUTPUT:\n{out[:2000]}\n\nWrite ONLY the system prompt:",
                    temperature=0.3,
                    max_tokens=512,
                ).strip()
            except Exception as exc:
                print(f"  [warn] pair {scene_id}: system prompt synthesis failed ({exc})", file=sys.stderr)
            item = {
                "id": scene_id,
                "chapter": f"Pair {scene_id}",
                "text": out,
                "chars": len(out),
                "input_prompt": inp,
                "system_prompt": sys_prompt,
            }
            append_jsonl(output, item)
            n_new += 1
            print(f"  pair {scene_id}/{len(io_matches) - 1}: input={len(inp)}c, output={len(out)}c, sys_prompt={len(sys_prompt)}c", flush=True)

        print(f"\nDone: {len(done) + n_new} total scenes ({n_new} new) -> {output}")
        return

    md_heading_re = build_header_regex(cfg)
    blocks = structural_blocks(text, chapter_regex(pack, full_line=True), md_heading_re)
    print(f"Structural pass: {len(blocks)} blocks")

    long_blocks = [b for b in blocks if len(b[1]) > segment["max_block_chars"]]
    if long_blocks:
        print(f"{len(long_blocks)} of them need the LLM pass "
              f"(longer than segment.max_block_chars = {segment['max_block_chars']})")

    debug_dir = work_path(cfg, "debug")
    collected = []
    for index, (chapter, block) in enumerate(blocks):
        if len(block) <= segment["max_block_chars"]:
            collected.append((chapter, block))
            continue
        paragraphs = paragraphs_of(block)
        print(f"Block {index} ({chapter or 'no chapter'}): {len(block)} chars, "
              f"{len(paragraphs)} paragraphs -> LLM")
        boundaries = llm_boundaries(llm, pack, paragraphs, segment["llm_window_chars"],
                                    segment["llm_window_overlap"], debug_dir, block_id=index)
        for scene in split_by_boundaries(paragraphs, boundaries):
            collected.append((chapter, scene))

    # Process (chapter, scene_text) tuples directly through merge and cap passes
    # Cap oversized scenes first, then merge undersized fragments, then enforce max ceiling
    max_cap = int(segment["max_scene_chars"])
    processed = cap_long(collected, max_cap)
    processed = merge_short(processed, segment["min_scene_chars"])
    processed = cap_long(processed, max_cap)
    assert processed, "segmentation produced no scenes — check the source file"

    items = []
    for scene_id, (chapter_title, scene_text) in enumerate(processed):
        items.append({
            "id": scene_id,
            "chapter": chapter_title,
            "text": scene_text,
            "chars": len(scene_text)
        })

    output = work_path(cfg, "scenes.jsonl")
    write_jsonl(output, items)

    lengths = sorted(item["chars"] for item in items)
    oversized = [item for item in items if item["chars"] > segment["max_scene_chars"]]
    
    trace_log = os.path.join(debug_dir, "stage_01_trace.log")
    with open(trace_log, "w", encoding="utf-8") as tf:
        for it in items:
            tf.write(f"[STAGE 01][SCENE] id={it['id']} chapter=\"{it['chapter']}\" chars={it['chars']}\n")

    print(f"\nDone: {len(items)} scenes -> {output}")
    if oversized:
        print(f"[!] {len(oversized)} scenes are still over segment.max_scene_chars "
              f"({segment['max_scene_chars']}), largest {max(i['chars'] for i in oversized)}. "
              f"They will be expensive in step 02 and may overflow the context.")
    print(f"Scene length: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}")
    print("\nCheckpoint: if the median is far below segment.min_scene_chars, the "
          "chunker is shredding the text — raise the thresholds and rerun.")


if __name__ == "__main__":
    main()
