# -*- coding: utf-8 -*-
"""Step 03 — turn each scene into a roleplay transcript.

The conversion rules, and why they are what they are:

  * char is the most talkative character of the scene, or the one named in
    convert.char. Its lines plus the surrounding narration become one turn.
  * The narration is carried over with minimal edits rather than retold. If the
    converter is allowed to paraphrase, the dataset ends up in the converter's
    voice instead of the source's, which defeats the whole exercise.
  * user turns are the interlocutor's lines only, with no narration. That is
    deliberate asymmetry: it teaches the model to answer richly to thin input,
    which is what a real chat session looks like.
  * In a crowd scene, everyone else becomes part of the world; their lines go
    inside char's narration.

    Input:  <workdir>/scenes_attr.jsonl
    Output: <workdir>/rp_raw.jsonl   {id, source_id, char, user, turns}

    python 03_convert.py [--config config.yaml]
"""
import json
import os
import re
import sys
from collections import Counter

from pipeline.axes import stable_rng
from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file, write_jsonl
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map


def clean_scene_text(text):
    """Strip image embeds, OCR noise artifacts, and collapse excessive spacing."""
    if not text:
        return ""
    # Remove markdown image embeds: ![Image](images/sample_img_1.png)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove long runs of dashes, pipes, underscores, or OCR form rules
    text = re.sub(r"[-|—_=~*]{3,}", " ", text)
    # Remove excessive blank checkboxes/OCR bullets
    text = re.sub(r"(?:^|\s)(?:O\s*O|0\s*0|\[\s*\]|\(s*\))(?:$|\s)", " ", text)
    # Collapse multiple blank lines & horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_usable_content(text):
    """Detect if a scene has real prose/words or is empty/corrupt."""
    if not text or len(text.strip()) < 80:
        return False, "too short (< 80 chars)"
    
    words = re.findall(r"\b[A-Za-zА-Яа-я]{2,}\b", text)
    if not words:
        return False, "no recognizable words"
    
    # Only skip if text is almost exclusively symbols or digits
    alpha_chars = sum(1 for c in text if c.isalpha())
    if (alpha_chars / max(1, len(text))) < 0.15:
        return False, "low alphabet character ratio (likely binary or pure table lines)"
        
    return True, "ok"


def repair_and_filter_scenes(cfg, debug_dir):
    """Scan scenes_attr.jsonl, repair corrupted lines, filter pure OCR junk, and write repaired dataset."""
    in_path = require_file(work_path(cfg, "scenes_attr.jsonl"), "02_attribute.py")
    out_path = work_path(cfg, "scenes_attr_repaired.jsonl")
    skipped_log = os.path.join(debug_dir, "skipped_garbage_scenes.jsonl")

    raw_items = []
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                raw_items.append(json.loads(line_str))
            except Exception:
                # Try simple brace extraction if line is wrapped with noise
                m = re.search(r"\{.*\}", line_str)
                if m:
                    try:
                        raw_items.append(json.loads(m.group(0)))
                    except Exception:
                        continue

    seen_ids = set()
    repaired, skipped = [], []

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        scene_id = item.get("id")
        if scene_id is None or scene_id in seen_ids:
            continue
        seen_ids.add(scene_id)

        # Standardize and clean fields
        cleaned_text = clean_scene_text(str(item.get("text", "")))
        usable, reason = is_usable_content(cleaned_text)
        if not usable:
            skipped.append({"id": scene_id, "reason": reason, "chars": len(cleaned_text), "preview": cleaned_text[:120]})
            continue

        item["text"] = cleaned_text
        item["chars"] = len(cleaned_text)
        item.setdefault("lines", [])
        item.setdefault("characters", [])
        item.setdefault("dialogue_share", 0.0)

        repaired.append(item)

    write_jsonl(out_path, repaired)
    if skipped:
        write_jsonl(skipped_log, skipped)
        print(f"  [prep] Scanned {len(raw_items)} scenes: {len(repaired)} clean scenes kept, {len(skipped)} non-narrative forms skipped -> {skipped_log}")
    else:
        print(f"  [prep] All {len(repaired)} scenes passed validation -> {out_path}")

    return out_path


def pick_pair(scene, forced_char, unknown, generic):
    """Choose (char, user) for a scene, or (None, None) if it is unusable.

    A speaker is only a candidate if it is a real name: not the unknown marker,
    not a pronoun or a role word, and longer than one character. Those come back
    from attribution regularly, and a transcript whose char is "the voice"
    teaches nothing.
    """
    speakers = []
    for line in scene.get("lines", []):
        name = str(line.get("speaker", "")).strip()
        if not name or len(name) <= 1:
            continue
        if name.lower() == unknown or name.lower() in generic:
            continue
        speakers.append(name)
    if not speakers:
        return None, None

    counts = Counter(speakers)
    if forced_char and forced_char.lower() != "auto":
        if forced_char not in counts:
            return None, None
        char = forced_char
    else:
        char = counts.most_common(1)[0][0]

    others = [name for name, _ in counts.most_common() if name != char]
    if not others:
        return None, None
    return char, others[0]  # user = the second most active speaker


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    convert = cfg["convert"]
    llm = LLM(cfg)
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)
    unknown = pack.unknown_speaker.lower()
    generic = pack.generic_speakers

    debug_dir = work_path(cfg, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    repaired_scenes_path = repair_and_filter_scenes(cfg, debug_dir)
    scenes = read_jsonl(repaired_scenes_path)

    out_path = work_path(cfg, "rp_raw.jsonl")
    refusal_path = work_path(cfg, "refusals_03.jsonl")

    existing = read_jsonl(out_path)
    done = {item["source_id"] for item in existing}
    next_id = len(existing)
    if done:
        print(f"[resume] {next_id} transcripts already written, continuing")

    pending, skipped = [], 0
    allow_informational = cfg.get("convert", {}).get("allow_informational", True)
    default_speaker = pack.get("meta.fallback_char", "Narrator")
    default_listener = pack.get("meta.fallback_user", "Inquirer")

    for scene in scenes:
        if scene["id"] in done:
            continue
        char, user = pick_pair(scene, convert.get("char", "auto"), unknown, generic)
        if not char or scene.get("dialogue_share", 0) < convert["min_dialogue_share"]:
            if allow_informational and len(scene.get("text", "")) >= 300:
                # Informational / non-dialogue scene: convert into a teacher/student exchange
                char, user = default_speaker, default_listener
                scene["_is_informational"] = True
            else:
                skipped += 1
                continue
        else:
            scene["_is_informational"] = False

        scene["_char"], scene["_user"] = char, user
        pending.append(scene)

    print(f"To convert: {len(pending)} scenes ({skipped} skipped as unusable)")
    if not pending:
        print("Nothing to do. If everything was skipped, check convert.char and "
              "convert.min_dialogue_share, and re-read the step 02 output.")
        return

    def work(scene):
        if scene.get("_is_informational"):
            # Determine anti-data-dump constraint mode: 'on', 'off', or 'random'
            anti_dump_mode = str(convert.get("anti_dump", "random")).strip().lower()
            if anti_dump_mode == "on":
                anti_dump_active = True
            elif anti_dump_mode == "off":
                anti_dump_active = False
            else:
                # Deterministic Bernoulli trial seeded by scene ID
                rng = stable_rng(cfg.get("augment", {}).get("seed", 420), "anti_dump", scene["id"])
                anti_dump_active = rng.random() > 0.50

            if anti_dump_active:
                rule_1 = (
                    f"1. NO DATA DUMPING: '{scene['_char']}' MUST NOT explain the entire passage in the first turn. "
                    f"Reveal facts organically and incrementally across turns as '{scene['_user']}' reacts or probes deeper."
                )
            else:
                rule_1 = (
                    f"1. NATURAL EXPOSITION: '{scene['_char']}' may speak expansively and share comprehensive details "
                    f"freely as fits their mood and expertise, without artificial pacing limits."
                )

            # Determine cinematic action weaving mode: 'on', 'off', or 'random'
            cinematic_mode = str(convert.get("cinematic_beats", "random")).strip().lower()
            if cinematic_mode == "on":
                cinematic_active = True
            elif cinematic_mode == "off":
                cinematic_active = False
            else:
                rng_cine = stable_rng(cfg.get("augment", {}).get("seed", 420), "cinematic_beats", scene["id"])
                cinematic_active = rng_cine.random() > 0.50

            if cinematic_active:
                rule_2 = (
                    f"2. CINEMATIC PROSE & ACTION BEATS: Every turn by '{scene['_char']}' MUST weave dialogue together with "
                    f"physical gestures, environmental interactions, pauses, tone shifts, and character mannerisms."
                )
            else:
                rule_2 = (
                    f"2. DIALOGUE-FOCUSED PROSE: Focus directly on crisp, authentic dialogue and verbal exchange. "
                    f"Avoid excessive physical stage directions, action tags, or purple prose unless essential."
                )

            system_prompt = (
                f"You are an expert creative dramatist and roleplay conversion engine.\n"
                f"Convert the provided factual/expository text into a living, multi-turn roleplay conversation "
                f"between '{scene['_user']}' (an engaged, curious, or skeptical conversational partner) "
                f"and '{scene['_char']}' (an atmospheric, knowledgeable character who inhabits this world and voice).\n\n"
                f"CRITICAL DRAMATIC RULES:\n"
                f"{rule_1}\n"
                f"{rule_2}\n"
                f"3. DYNAMIC INTERLOCUTOR: '{scene['_user']}' must never sound like a textbook quiz or interviewer. "
                f"They should react with emotion, pushback, curiosity, surprise, or skepticism.\n"
                f"4. PRESERVE FACTUAL INTEGRITY: All historical names, terms, rituals, dates, and claims from the text "
                f"must be naturally spoken and discussed, never omitted or distorted.\n"
                f"5. OUTPUT FORMAT: Return valid JSON with alternating turns:\n"
                f"{{\"turns\": [{{\"role\": \"user\", \"text\": \"...\"}}, {{\"role\": \"char\", \"text\": \"...\"}}]}}"
            )
            user_prompt = (
                f"Source Passage to Dramatize:\n---\n{scene['text']}\n---\n"
                f"Characters:\n- Speaker / Expert: {scene['_char']}\n- Counterpart: {scene['_user']}\n\n"
                f"Produce a rich, progressive multi-turn roleplay transcript."
            )
            return llm.chat(system_prompt, user_prompt, temperature=convert["temperature"], json_mode=True)

        attribution = "; ".join(
            f"\"{line.get('quote', '')}...\" — {line.get('speaker', '')}"
            for line in scene.get("lines", []))
        return llm.chat(
            pack.require("convert.system"),
            render(pack.require("convert.user"),
                   scene=scene["text"], char=scene["_char"],
                   user=scene["_user"], attribution=attribution),
            temperature=convert["temperature"],
            json_mode=True,
        )

    debug_dir = work_path(cfg, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    n_refusals = 0

    for scene, raw, error in parallel_map(work, pending, workers):
        if error is not None:
            print(f"  [warn] scene {scene['id']}: {error} — will retry on the next run",
                  file=sys.stderr)
            continue

        data = None
        try:
            data = extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            if is_refusal(raw):
                append_jsonl(refusal_path, {"source_id": scene["id"], "head": raw[:300]})
                n_refusals += 1
                print(f"scene {scene['id']}: model refused -> {refusal_path}, not marked done")
                continue

            # Secondary aggressive salvage for convert transcripts:
            # 1. Standard pattern: {"role": "char", "text": "..."}
            turn_blocks = re.findall(
                r'["\']role["\']\s*:\s*["\'](user|char)["\']\s*,\s*["\']text["\']\s*:\s*["\'](.*?)(?<!\\)["\']\s*\}',
                raw,
                re.DOTALL
            )
            # 2. Reverse order: {"text": "...", "role": "char"}
            if not turn_blocks:
                rev_blocks = re.findall(
                    r'["\']text["\']\s*:\s*["\'](.*?)(?<!\\)["\']\s*,\s*["\']role["\']\s*:\s*["\'](user|char)["\']\s*\}',
                    raw,
                    re.DOTALL
                )
                if rev_blocks:
                    turn_blocks = [(role, txt) for (txt, role) in rev_blocks]

            # 3. Tagged turns fallback: [char]: text ... [user]: text ...
            if not turn_blocks:
                tag_blocks = re.findall(r'\[(char|user)\]\s*[:\-]?\s*(.*?)(?=\n\s*\[(?:char|user)\]|\Z)', raw, re.DOTALL | re.IGNORECASE)
                if tag_blocks:
                    turn_blocks = [(role.lower(), txt) for (role, txt) in tag_blocks]

            if turn_blocks:
                salvaged = []
                for role, txt in turn_blocks:
                    clean_txt = (
                        txt.replace('\\"', '"')
                           .replace('\\n', '\n')
                           .replace('\\t', '\t')
                           .strip()
                    )
                    if clean_txt:
                        salvaged.append({"role": role, "text": clean_txt})
                if salvaged:
                    data = {"turns": salvaged}

            if data is None:
                fail_file = os.path.join(debug_dir, f"convert_fail_{scene['id']}.txt")
                with open(fail_file, "w", encoding="utf-8") as f:
                    f.write(f"--- ERROR: {exc} ---\n\n" + (raw or ""))
                print(f"  [warn] scene {scene['id']}: invalid JSON (logged to {fail_file})",
                      file=sys.stderr)
                continue

        raw_turns = []
        if isinstance(data, list):
            raw_turns = data
        elif isinstance(data, dict):
            raw_turns = data.get("turns", [])

        turns = [
            turn for turn in raw_turns
            if isinstance(turn, dict)
            and turn.get("role") in ("char", "user")
            and str(turn.get("text", "")).strip()
        ]

        if len(turns) < convert["min_turns"] or not any(t["role"] == "char" for t in turns):
            fail_file = os.path.join(debug_dir, f"convert_dropped_{scene['id']}.txt")
            with open(fail_file, "w", encoding="utf-8") as f:
                f.write(f"--- REASON: Insufficient turns ({len(turns)} < {convert['min_turns']}) ---\n\n" + (raw or ""))
            skipped += 1
            continue

        append_jsonl(out_path, {"id": next_id, "source_id": scene["id"],
                                "char": scene["_char"], "user": scene["_user"],
                                "turns": turns})
        next_id += 1
        print(f"scene {scene['id']} -> {len(turns)} turns "
              f"({scene['_char']} / {scene['_user']})")

    print(f"\nDone: {next_id} transcripts, {skipped} skipped -> {out_path}")
    if n_refusals:
        print(f"{n_refusals} refusals. Rerun with a different llm.base_url to pick up only those.")
    print("Checkpoint: read 5-10 transcripts side by side with their source scenes. "
          "If the narration has been retold rather than carried over, fix the "
          "convert prompt in the language pack before running step 04.")


if __name__ == "__main__":
    main()
