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


_REPEAT_EXEMPT_WORDS = frozenset(
    "a an the and but or if so then than that this these those it its "
    "i you he she we they me my your his her our their "
    "is was are were be been am do does did done have has had "
    "not no nor yes yeah yep nah nope to of in on at for with from by as "
    "very really quite too also still even just only much many "
    "all some any every each oh ah ahh ohh ha haha heh hey hi hello bye "
    "hmm mm uh um er erm well now right okay ok sure fine good bad better best "
    "wow yay please wait stop more again go come back what why how when where who which".split()
)
_REPEATED_SENT_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]*)([.!?])(?:\s+\1\2)+", re.I)
_REPEATED_COMMA_END_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]*)(?:[,;]\s*\1\b)+\s*([.!?])", re.I)
_REPEATED_COMMA_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]*)(?:[,;]\s*\1\b)+", re.I)
_REPEATED_BARE_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]*)(?:\s+\1\b)+", re.I)


def collapse_repeated_words(text):
    """Collapse stalled word repetitions ('Word. Word.' -> 'Word.') while sparing emphasis."""
    if not text:
        return text

    def _keep_punct(m):
        w, p = m.group(1), m.group(2)
        return m.group(0) if w.lower() in _REPEAT_EXEMPT_WORDS else w + p

    def _keep_word(m):
        w = m.group(1)
        return m.group(0) if w.lower() in _REPEAT_EXEMPT_WORDS else w

    text = _REPEATED_SENT_RE.sub(_keep_punct, text)
    text = _REPEATED_COMMA_END_RE.sub(_keep_punct, text)
    text = _REPEATED_COMMA_RE.sub(_keep_word, text)
    text = _REPEATED_BARE_RE.sub(_keep_word, text)
    return text


def normalize_quotes(text):
    """Collapse doubled quotes and balance curly double quotes."""
    if not text:
        return text
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r"\u201c{2,}", "\u201c", text)
    text = re.sub(r"\u201d{2,}", "\u201d", text)
    n_open, n_close = text.count("\u201c"), text.count("\u201d")
    if n_open > n_close:
        text += "\u201d" * (n_open - n_close)
    elif n_close > n_open:
        text = "\u201c" * (n_close - n_open) + text
    return text


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


def chunk_prose_to_progressive_turns(text, chapter_name="", user_probes=None):
    """Deterministically transform raw prose into a high-density, multi-turn dialogue."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    turns = []
    title_ref = f" regarding {chapter_name}" if chapter_name else ""
    turns.append({
        "role": "user",
        "text": f"*Looks over{title_ref}.* Go on—tell me what happened."
    })

    default_probes = [
        "What happened next?",
        "Go on. Tell me the rest.",
        "And how did things unfold from there?",
        "What did you make of that?",
        "And then what?",
        "How did it end?",
        "*Listens closely as the account unfolds, leaning forward.* What occurred next?",
        "Gestures toward the map.* And how did the lines hold against that onslaught?",
        "*Eyes narrowing in astonishment.* How did the command respond to such odds?",
        "*Nods solemnly.* And what was the outcome of that clash?",
        "*A tense silence falls between them.* What fate befell those who stood their ground?",
        "*Inquires softly.* And what remained when the smoke finally cleared?"
    ]
    probes = user_probes if user_probes else default_probes

    for idx, para in enumerate(paragraphs):
        clean_para = para.strip()
        if not clean_para:
            continue
        turns.append({"role": "char", "text": clean_para})
        if idx < len(paragraphs) - 1:
            turns.append({"role": "user", "text": probes[idx % len(probes)]})

    return turns


def pick_pair(scene, forced_char, unknown, generic, invalid_set=None, fallback_char="The Chronicler", fallback_user="The Inquirer"):
    """Choose (char, user) for a scene, ensuring single-speaker journals are retained."""
    speakers = []
    invalid_set = invalid_set or set() # Words that are common nouns, verbs, or locations hallucinated as speaker names
    
    def _is_valid_name(n):
        s = str(n).strip()
        if not s or len(s) <= 1:
            return False
        low = s.lower()
        if low == unknown or low in invalid_set:
            return False
        if low in generic and low not in ("narrator", "chronicler"):
            return False
        return True

    for line in scene.get("lines", []):
        name = str(line.get("speaker", "")).strip()
        if _is_valid_name(name):
            speakers.append(name)

    # If no lines were attributed, check scene characters
    if not speakers and scene.get("characters"):
        speakers = [c for c in scene["characters"] if _is_valid_name(c)]

    # Single-speaker or solo-journal fallback
    if not speakers:
        return fallback_char, fallback_user

    counts = Counter(speakers)
    if forced_char and forced_char.lower() != "auto":
        char = forced_char
    else:
        char = counts.most_common(1)[0][0]

    others = [name for name, _ in counts.most_common() if name != char]
    user = others[0] if others else fallback_user
    return char, user


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
    default_speaker = pack.get("meta.fallback_char", "The Chronicler")
    default_listener = pack.get("meta.fallback_user", "The Inquirer")

    # Combine config invalid_speakers into a set for filtering
    raw_invalid = convert.get("invalid_speakers", [])
    if isinstance(raw_invalid, str):
        raw_invalid = [w.strip() for w in raw_invalid.split(",") if w.strip()]
    invalid_set = {str(w).strip().lower() for w in raw_invalid if w}

    source_mode = str(convert.get("source_mode", "auto")).strip().lower()

    for scene in scenes:
        if scene["id"] in done:
            continue
        char, user = pick_pair(
            scene, convert.get("char", "auto"), unknown, generic,
            invalid_set=invalid_set,
            fallback_char=default_speaker, fallback_user=default_listener
        )
        
        if source_mode == "informational":
            scene["_is_informational"] = True
        elif source_mode == "fiction":
            scene["_is_informational"] = False
        else:
            # Auto-detect: informational if no secondary speaker or very low dialogue
            if not char or user == default_listener or len(scene.get("lines", [])) <= 2:
                scene["_is_informational"] = True
            else:
                scene["_is_informational"] = False

        # In v5, single-speaker logs and lore entries are never discarded
        if not char:
            char, user = default_speaker, default_listener

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
        user_prompt = render(
            pack.require("convert.user"),
            scene=scene["text"], char=scene["_char"],
            user=scene["_user"], attribution=attribution
        )
        # For long prose chapters, append explicit instruction against truncation
        if len(scene.get("text", "")) > 1500:
            user_prompt += (
                f"\n\nCRITICAL: Do NOT summarize or truncate. Carry over the full narration, atmospheric descriptions, "
                f"and events from the scene into the dialogue turns."
            )

        return llm.chat(
            pack.require("convert.system"),
            user_prompt,
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
                print(f"  [warn] scene {scene['id']}: invalid JSON (logged to {fail_file}); triggering fallback builder",
                      file=sys.stderr)

        raw_turns = []
        if isinstance(data, list):
            # If model returned [{"turns": [...]}] or [{"turns": [...]}, {"turns": []}]
            if data and isinstance(data[0], dict) and "turns" in data[0]:
                for item in data:
                    if isinstance(item, dict) and isinstance(item.get("turns"), list):
                        raw_turns.extend(item["turns"])
            else:
                raw_turns = data
        elif isinstance(data, dict):
            raw_turns = data.get("turns", [])

        # Filter out trailing hallucinated integer/token loops like [1], [2], [3]...
        clean_raw = []
        for t in raw_turns:
            if isinstance(t, (int, float)):
                continue
            if isinstance(t, list) and len(t) == 1 and isinstance(t[0], (int, float)):
                continue
            clean_raw.append(t)
        raw_turns = clean_raw

        turns = []
        for i, turn in enumerate(raw_turns):
            if isinstance(turn, dict):
                # Pattern A: Standard {"role": "char", "text": "..."}
                role = turn.get("role")
                text_val = str(turn.get("text") or turn.get("turn") or "").strip()
                
                # Pattern B: {"char": "text"} or {"user": "text"}
                if not text_val:
                    if "char" in turn and str(turn["char"]).strip():
                        role = "char"
                        text_val = str(turn["char"]).strip()
                    elif "user" in turn and str(turn["user"]).strip():
                        role = "user"
                        text_val = str(turn["user"]).strip()
                    else:
                        # Pattern C: Dict keys as dialogue statements (e.g. convert_dropped_49)
                        for k, v in turn.items():
                            k_str, v_str = str(k).strip(), str(v).strip()
                            if len(k_str) > 20 and not k_str.startswith("{"):
                                turns.append({"role": "char", "text": k_str})
                            if len(v_str) > 20 and v is not None and str(v).lower() != "none":
                                turns.append({"role": "user", "text": v_str})
                        continue

                if not text_val:
                    continue
                if role not in ("char", "user"):
                    role = "char" if i % 2 == 0 else "user"
                text_val = normalize_quotes(collapse_repeated_words(text_val))
                turns.append({"role": role, "text": text_val})
            elif isinstance(turn, str) and turn.strip():
                role = "user" if i % 2 == 0 else "char"
                turns.append({"role": role, "text": turn.strip()})

        # Ensure conversation starts with user if it's an informational dialogue
        if turns and turns[0]["role"] == "char" and len(turns) == 1:
            ch_name = scene.get("chapter", "the campaign")
            opener = f"*Inquires about the events recorded in {ch_name}. What transpired on the field?*"
            turns.insert(0, {"role": "user", "text": opener})

        # High-Fidelity Guarantee:
        # If model returned no character turns, heavily truncated (< 65% of source), or looped,
        # build a complete multi-turn dialogue directly from the source prose.
        char_text = " ".join(t["text"] for t in turns if t.get("role") == "char")
        source_text = clean_scene_text(scene.get("text", ""))

        has_repetition_loop = False
        if len(turns) > 8:
            turn_texts = [t["text"][:60] for t in turns]
            if len(turn_texts) - len(set(turn_texts)) > 3:
                has_repetition_loop = True

        trunc_thresh = float(convert.get("truncation_threshold", 0.65))
        is_truncated = len(source_text) > 600 and (len(char_text) < len(source_text) * trunc_thresh)

        used_fallback = False
        if not any(t["role"] == "char" for t in turns) or has_repetition_loop or is_truncated:
            if convert.get("fallback_on_fail", True):
                ch_name = scene.get("chapter") or ""
                turns = chunk_prose_to_progressive_turns(source_text, chapter_name=ch_name)
                used_fallback = True
            elif not any(t["role"] == "char" for t in turns):
                print(f"scene {scene['id']}: skipped (no char turns, fallback_on_fail is false)")
                skipped += 1
                continue

        append_jsonl(out_path, {"id": next_id, "source_id": scene["id"],
                                "char": scene["_char"], "user": scene["_user"],
                                "turns": turns})
        mode_tag = " [fallback]" if used_fallback else ""
        print(f"scene {scene['id']} -> rp {next_id}: {len(turns)} turns "
              f"({scene['_char']} / {scene['_user']}){mode_tag}")
        next_id += 1

    print(f"\nDone: {next_id} transcripts, {skipped} skipped -> {out_path}")
    if n_refusals:
        print(f"{n_refusals} refusals. Rerun with a different llm.base_url to pick up only those.")
    print("Checkpoint: read 5-10 transcripts side by side with their source scenes. "
          "If the narration has been retold rather than carried over, fix the "
          "convert prompt in the language pack before running step 04.")


if __name__ == "__main__":
    main()
