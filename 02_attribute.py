# -*- coding: utf-8 -*-
"""Step 02 — work out who is in each scene and who says which line.

This is a separate step rather than part of the conversion because attribution
is the single largest source of errors downstream, and it has to be inspectable
before it disappears into a transcript. Spot-check a dozen scenes here and you
save yourself rerunning everything after step 04.

The step is resumable: results are appended one scene at a time, and a rerun
processes only what is missing. A model refusal is recorded but NOT marked done,
so pointing the config at a different model and rerunning picks up exactly the
refused scenes.

    Input:  <workdir>/scenes.jsonl
    Output: <workdir>/scenes_attr.jsonl
            {..., characters, lines, dialogue_share, unknown_speakers}

    python 02_attribute.py [--config config.yaml]
"""
import os
import re
import sys

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    llm = LLM(cfg)
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)
    unknown = pack.unknown_speaker.lower()

    scenes = read_jsonl(require_file(work_path(cfg, "scenes.jsonl"), "01_segment.py"))
    out_path = work_path(cfg, "scenes_attr.jsonl")
    refusal_path = work_path(cfg, "refusals_02.jsonl")

    done = {item["id"] for item in read_jsonl(out_path)}
    pending = [scene for scene in scenes if scene["id"] not in done]
    if done:
        print(f"[resume] {len(done)} already done, {len(pending)} left")
    if not pending:
        print("Nothing to do. Delete the output file to redo this step from scratch.")
        return

    debug_dir = work_path(cfg, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    trace_log = os.path.join(debug_dir, "stage_02_trace.log")

    use_colon_dialogue = cfg.get("attribute", {}).get("colon_dialogue", True)
    COLON_SPEECH_RE = re.compile(r'([A-Z][A-Za-z0-9\s’\'-]{1,35}?)\s*:\s*\n\s*(["“«][^"”»\n]+["”»])')

    raw_invalid = cfg.get("convert", {}).get("invalid_speakers", [])
    if isinstance(raw_invalid, str):
        raw_invalid = [w.strip() for w in raw_invalid.split(",") if w.strip()]
    invalid_set = {str(w).strip().lower() for w in raw_invalid if w}
    invalid_set |= set(pack.generic_speakers)
    invalid_set.add(unknown)

    def work(scene):
        user_prompt = render(pack.require("attribute.user"), scene=scene["text"])
        if use_colon_dialogue:
            colon_matches = COLON_SPEECH_RE.findall(scene["text"])
            if colon_matches:
                examples_hint = "\nNotice: The text contains action/dialogue pairs separated by colons and newlines. Example lines found:\n"
                for speaker_candidate, quote_candidate in colon_matches[:4]:
                    clean_spk = re.sub(r'^(?:An?|The)\s+', '', speaker_candidate.strip())
                    examples_hint += f'- Speaker: "{clean_spk}", Quote: "{quote_candidate[:40]}..."\n'
                user_prompt += examples_hint

        return llm.chat(
            pack.require("attribute.system"),
            user_prompt,
            temperature=cfg["attribute"]["temperature"],
            json_mode=True,
        )

    n_refusals = 0
    for scene, raw, error in parallel_map(work, pending, workers):
        if error is not None:
            print(f"  [warn] scene {scene['id']}: {error} — will retry on the next run",
                  file=sys.stderr)
            continue

        try:
            data = extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            if is_refusal(raw):
                append_jsonl(refusal_path, {"id": scene["id"], "head": raw[:300]})
                n_refusals += 1
                print(f"scene {scene['id']}: model refused -> {refusal_path}, not marked done")
                continue

            # Fallback heuristic: parse {"characters": [...]} or {"lines": [...]} via regex
            data = None
            char_match = re.search(r'["\']characters["\']\s*:\s*\[(.*?)\]', raw, re.DOTALL)
            found_chars = []
            if char_match:
                found_chars = [c.strip().strip('"\'') for c in char_match.group(1).split(",") if c.strip().strip('"\'')]

            line_matches = re.findall(
                r'\{\s*["\']speaker["\']\s*:\s*["\'](.*?)["\']\s*,\s*["\']quote["\']\s*:\s*["\'](.*?)["\']\s*\}',
                raw,
                re.DOTALL
            )
            found_lines = [{"speaker": spk.strip(), "quote": q.strip()} for spk, q in line_matches]

            if found_chars or found_lines:
                data = {"characters": found_chars, "lines": found_lines}
            else:
                fail_file = os.path.join(debug_dir, f"attr_fail_{scene['id']}.txt")
                with open(fail_file, "w", encoding="utf-8") as f:
                    f.write(f"--- ERROR: {exc} ---\n\n" + (raw or ""))
                print(f"  [warn] scene {scene['id']}: invalid JSON (logged to {fail_file})",
                      file=sys.stderr)
                continue

        # Handle polymorphic returns: dict or list of dicts
        payload = data[0] if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) else data
        if not isinstance(payload, dict):
            payload = {}

        lines = [line for line in payload.get("lines", []) if isinstance(line, dict)]
        raw_characters = payload.get("characters", [])
        if not isinstance(raw_characters, list):
            raw_characters = [str(raw_characters)] if raw_characters else []

        # Deduplicate characters preserving order and filter out generic 'unknown' markers
        seen_chars = set()
        characters = []
        for c in raw_characters:
            c_str = str(c).strip().strip('"\'')
            if not c_str or len(c_str) <= 2:
                continue
            c_lower = c_str.lower()
            if c_lower in invalid_set or c_lower == "unknown" or c_lower in seen_chars:
                continue
            seen_chars.add(c_lower)
            characters.append(c_str)

        # Heuristic fallback: if model missed colon-break dialogue, extract it directly
        if not lines:
            for speaker_candidate, quote_candidate in COLON_SPEECH_RE.findall(scene["text"]):
                clean_spk = re.sub(r'^(?:An?|The)\s+', '', speaker_candidate.strip()).split()[-1]
                lines.append({"speaker": clean_spk, "quote": quote_candidate[:50]})
                if clean_spk.lower() not in seen_chars and clean_spk.lower() != unknown:
                    seen_chars.add(clean_spk.lower())
                    characters.append(clean_spk)

        # Monologue / Journal entry fallback: attribute to The Chronicler / Narrator
        if not characters:
            if re.search(r'\b(?:I\b|my\b|we\b|our\b|journal\b|camp\b|scout\b)', scene["text"], re.I):
                characters = ["The Chronicler"]

        scene["characters"] = characters
        scene["lines"] = lines

        # Rough share of the scene taken up by direct speech.
        quoted = sum(len(line.get("quote", "")) for line in lines)
        if lines:
            scene["dialogue_share"] = round(min(1.0, max(0.15, quoted * 4 / max(1, scene["chars"]))), 2)
        else:
            # Informational / monologue journal scene
            scene["dialogue_share"] = 0.05 if len(scene["text"]) > 200 else 0.0

        scene["unknown_speakers"] = sum(
            1 for line in lines if str(line.get("speaker", "")).lower() == unknown)

        append_jsonl(out_path, scene)
        print(f"scene {scene['id']}: {len(scene['characters'])} characters, "
              f"{len(lines)} lines, {scene['unknown_speakers']} unattributed")
        with open(trace_log, "a", encoding="utf-8") as tf:
            tf.write(f"[STAGE 02][ATTR] id={scene['id']} characters={scene['characters']} lines={len(lines)} dialogue_share={scene['dialogue_share']}\n")

    print(f"\nDone -> {out_path}")
    if n_refusals:
        print(f"{n_refusals} refusals. Point llm.base_url at a different model and rerun; "
              f"resume will process only the refused scenes.")
    print("Checkpoint: open 10-15 scenes with a non-zero unknown_speakers and check the "
          "attribution by eye. Everything after this step trusts it.")


if __name__ == "__main__":
    main()
