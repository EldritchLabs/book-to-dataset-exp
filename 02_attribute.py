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

    def work(scene):
        return llm.chat(
            pack.require("attribute.system"),
            render(pack.require("attribute.user"), scene=scene["text"]),
            temperature=cfg["attribute"]["temperature"],
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
        characters = payload.get("characters", [])
        if not isinstance(characters, list):
            characters = [str(characters)] if characters else []

        scene["characters"] = characters
        scene["lines"] = lines

        # Rough share of the scene taken up by direct speech. The quote field
        # holds only the opening words of each line, so it is multiplied by a
        # constant to approximate the full line. Used as a cheap filter in step
        # 03, never as a precise measurement.
        quoted = sum(len(line.get("quote", "")) for line in lines)
        scene["dialogue_share"] = round(min(1.0, quoted * 4 / max(1, scene["chars"])), 2)
        scene["unknown_speakers"] = sum(
            1 for line in lines if str(line.get("speaker", "")).lower() == unknown)

        append_jsonl(out_path, scene)
        print(f"scene {scene['id']}: {len(scene['characters'])} characters, "
              f"{len(lines)} lines, {scene['unknown_speakers']} unattributed")

    print(f"\nDone -> {out_path}")
    if n_refusals:
        print(f"{n_refusals} refusals. Point llm.base_url at a different model and rerun; "
              f"resume will process only the refused scenes.")
    print("Checkpoint: open 10-15 scenes with a non-zero unknown_speakers and check the "
          "attribution by eye. Everything after this step trusts it.")


if __name__ == "__main__":
    main()
