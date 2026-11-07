# -*- coding: utf-8 -*-
"""Step 05 — generate synthetic sessions anchored to the source.

Why this exists. A dataset built only from converted scenes teaches the model to
reproduce those scenes. Ask it for a situation the source does not contain and
it will quote the nearest one it knows. Synthetic rows are the counterweight:
new situations, written by the model in its own words, but with the characters
reasoning the way the source's characters reason.

Every generation is anchored by two or three real scenes, shown to the model as
a sample of how the character thinks, not as prose to imitate. Variety comes
from a situation matrix crossed with a character rotation and a random anchor
draw, all seeded by row id, so a resumed run produces the same rows it would
have produced without the interruption.

Three filters run as rows arrive:

  refusal      recorded separately, not marked done, picked up by a rerun
  anti-copy    overlap with the anchors above the threshold means the model
               rewrote the source instead of inventing
  dedupe       4-gram Jaccard against accepted rows, plus a stock-phrase check
               that catches skeletons reused across many rows

    Input:  <workdir>/scenes_attr.jsonl
    Output: <workdir>/synth_clean.jsonl (same shape as rp_clean.jsonl)
            <workdir>/synth_rejected.jsonl

    python 05_synthesize.py [--config config.yaml]
"""
import os
import random
import re
import sys
from collections import Counter

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map
from pipeline.textstats import jaccard, ngrams, overlap

# Turn count ranges, drawn per row. A dataset where every session is the same
# length teaches that length.
TURN_RANGES = [(2, 3), (3, 4), (4, 6), (6, 8)]


def synthesize_missing_turn(llm, turns, char, situation, anchor_text):
    """If a generation produced only 1 turn, synthesize the missing counterpart."""
    if len(turns) != 1:
        return turns

    sole_turn = turns[0]
    role = sole_turn.get("role")
    text = sole_turn.get("text", "").strip()
    if not text:
        return turns

    if role == "char":
        # Missing user opening: generate a fitting user prompt/action
        sys_prompt = (
            f"You are a roleplay dialogue assistant. Given a character's response from '{char}' "
            f"and the setting/situation, write ONLY the preceding interlocutor ('user') turn "
            f"that naturally prompted this response.\n"
            f"Keep it concise (1-2 sentences), conversational, emotional, or inquisitive. "
            f"Output plain text only with no prefixes or quotes."
        )
        user_prompt = (
            f"Situation: {situation}\n\n"
            f"Character's statement ({char}):\n\"{text}\"\n\n"
            f"Write the opening user turn that prompted this:"
        )
        try:
            raw_user = llm.chat(sys_prompt, user_prompt, temperature=0.7, max_tokens=256)
            clean_user = raw_user.strip().strip('"').strip("'")
            if clean_user:
                return [{"role": "user", "text": clean_user}, sole_turn]
        except Exception:
            pass

    elif role == "user":
        # Missing character response: generate char turn
        sys_prompt = (
            f"You are roleplaying as '{char}'. Ground your voice in the provided style anchors. "
            f"Respond directly to the user's turn in 2-4 sentences, weaving physical actions and dialogue.\n"
            f"Output plain text only with no prefixes."
        )
        user_prompt = (
            f"Style Anchors:\n{anchor_text[:2000]}\n\n"
            f"Situation: {situation}\n\n"
            f"User says:\n\"{text}\"\n\n"
            f"Generate {char}'s response:"
        )
        try:
            raw_char = llm.chat(sys_prompt, user_prompt, temperature=0.7, max_tokens=512)
            clean_char = raw_char.strip().strip('"').strip("'")
            if clean_char:
                return [sole_turn, {"role": "char", "text": clean_char}]
        except Exception:
            pass

    return turns


def top_characters(scenes, unknown, generic, limit, min_lines=5):
    """Characters worth rotating through: real names with enough lines."""
    counts = Counter()
    for scene in scenes:
        for line in scene.get("lines", []):
            name = str(line.get("speaker", "")).strip()
            if len(name) <= 1 or name.lower() == unknown or name.lower() in generic:
                continue
            counts[name] += 1
    return [name for name, count in counts.most_common(limit * 2)
            if count >= min_lines][:limit]


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    synth = cfg["synth"]
    if not synth["enabled"]:
        print("synth.enabled is false — nothing to do. "
              "Step 06 will build the dataset from the converted scenes only.")
        return

    pack = load_pack(cfg["lang"])
    llm = LLM(cfg)
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)

    situations = pack.require("synth.situations")
    behavior_mods = pack.get("synth.behavior_mods", [""])
    assert situations, "the language pack has an empty synth.situations list"

    scenes = read_jsonl(require_file(work_path(cfg, "scenes_attr.jsonl"), "02_attribute.py"))
    characters = synth.get("chars") or top_characters(
        scenes, pack.unknown_speaker.lower(), pack.generic_speakers, synth["chars_limit"])

    if not characters:
        # Fallback 1: Collect any named entities in scene['characters']
        candidates = Counter()
        for s in scenes:
            for c in s.get("characters", []):
                name = str(c).strip()
                if len(name) > 1 and name.lower() != pack.unknown_speaker.lower() and name.lower() not in pack.generic_speakers:
                    candidates[name] += 1
        if candidates:
            characters = [name for name, _ in candidates.most_common(synth["chars_limit"])]

    if not characters:
        # Fallback 2: Fall back to language pack defaults
        characters = [pack.fallback_char]

    print(f"Character rotation: {', '.join(characters)}")

    # Anchor scenes per character: the ones where they actually speak. Falling
    # back to all scenes keeps a rarely-speaking character usable.
    anchors_by_char = {
        name: [s for s in scenes if any(l.get("speaker") == name for l in s.get("lines", []))] or scenes
        for name in characters
    }

    out_path = work_path(cfg, "synth_clean.jsonl")
    reject_path = work_path(cfg, "synth_rejected.jsonl")
    refusal_path = work_path(cfg, "refusals_05.jsonl")

    accepted = read_jsonl(out_path)
    done = {item["id"] for item in accepted}
    done |= {item["id"] for item in read_jsonl(reject_path)}

    # n-gram sets of accepted rows, for the dedupe check, plus their document
    # frequency, for the stock-phrase check.
    accepted_grams = [ngrams("\n".join(t["text"] for t in item["turns"])) for item in accepted]
    gram_df = Counter()
    for grams in accepted_grams:
        gram_df.update(grams)
    if done:
        print(f"[resume] {len(done)} rows already processed, continuing")

    def ingredients(row_id):
        """Everything one row is built from, decided by its id alone."""
        rng = random.Random(row_id)
        char = characters[row_id % len(characters)]
        pool = anchors_by_char[char]
        picked = rng.sample(pool, min(synth["anchors_per_gen"], len(pool)))
        anchor_text = "\n---\n".join(a["text"][:synth["anchor_max_chars"]] for a in picked)
        turns_min, turns_max = rng.choice(TURN_RANGES)
        return char, rng.choice(situations), anchor_text, turns_min, turns_max, rng.choice(behavior_mods)

    pending = [i for i in range(synth["count"]) if i not in done]
    print(f"To generate: {len(pending)} rows")
    if not pending:
        print("Nothing to do. Raise synth.count to generate more.")
        return

    def work(row_id):
        char, situation, anchor_text, turns_min, turns_max, modifier = ingredients(row_id)
        prompt = render(pack.require("synth.user"), char=char, anchors=anchor_text,
                        situation=situation, turns_min=turns_min, turns_max=turns_max)
        if modifier:
            prompt += "\n" + modifier
        system = render(pack.require("synth.system"), char=char)
        raw = llm.chat(system, prompt, temperature=synth["temperature"], json_mode=True)
        return raw, char, anchor_text, turns_min, situation

    debug_dir = work_path(cfg, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    n_ok = n_rejected = n_refusals = 0
    for row_id, result, error in parallel_map(work, pending, workers):
        if error is not None:
            print(f"  [warn] row {row_id}: {error} — will retry on the next run", file=sys.stderr)
            continue
        raw, char, anchor_text, turns_min, situation = result
        data = None

        try:
            data = extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            if is_refusal(raw):
                append_jsonl(refusal_path, {"id": row_id, "head": raw[:300]})
                n_refusals += 1
                print(f"row {row_id}: model refused — not marked done")
                continue

            # Turn-level salvage for synthetic rows
            turn_blocks = re.findall(
                r'["\']role["\']\s*:\s*["\'](user|char)["\']\s*,\s*["\']text["\']\s*:\s*["\'](.*?)["\']\s*\}',
                raw,
                re.DOTALL
            )
            if not turn_blocks:
                turn_blocks = re.findall(r'\[(char|user)\]\s*[:\-]?\s*(.*?)(?=\n\s*\[(?:char|user)\]|\Z)', raw, re.DOTALL | re.IGNORECASE)

            if turn_blocks:
                salvaged = []
                for role, txt in turn_blocks:
                    clean_txt = txt.replace('\\"', '"').replace('\\n', '\n').strip()
                    if clean_txt:
                        salvaged.append({"role": role.lower(), "text": clean_txt})
                if salvaged:
                    data = {"turns": salvaged}

            if data is None:
                fail_file = os.path.join(debug_dir, f"synth_fail_{row_id}.txt")
                with open(fail_file, "w", encoding="utf-8") as f:
                    f.write(f"--- ERROR: {exc} ---\n\n" + (raw or ""))
                print(f"  [warn] row {row_id}: invalid JSON (logged to {fail_file})", file=sys.stderr)
                continue

        raw_turns = data.get("turns", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        turns = [t for t in raw_turns
                 if isinstance(t, dict) and t.get("role") in ("user", "char")
                 and str(t.get("text", "")).strip()]

        # Auto-complete single-turn generations (Q without A, or A without Q)
        if len(turns) == 1:
            turns = synthesize_missing_turn(llm, turns, char, situation, anchor_text)

        text = "\n".join(t["text"] for t in turns)

        def reject(reason):
            append_jsonl(reject_path, {"id": row_id, "reject_reason": reason})
            print(f"row {row_id}: REJECT {reason}")

        min_turns = min(turns_min, 2)
        if len(turns) < min_turns or not any(t["role"] == "char" for t in turns):
            reject(f"too few turns: {len(turns)} < {min_turns}")
            n_rejected += 1
            continue

        anchor_overlap = round(overlap(text, anchor_text), 3)
        if anchor_overlap > synth["max_anchor_overlap"]:
            reject(f"copies the anchors: overlap {anchor_overlap} > {synth['max_anchor_overlap']}")
            n_rejected += 1
            continue

        grams = ngrams(text)
        similarity = max((jaccard(grams, other) for other in accepted_grams), default=0.0)
        if similarity > synth["dedupe_jaccard"]:
            reject(f"duplicate: jaccard {similarity:.2f} > {synth['dedupe_jaccard']}")
            n_rejected += 1
            continue

        # Stock skeletons: n-grams that already appear in a large share of the
        # accepted rows. Two hits mean the model has settled into a template,
        # which a pairwise duplicate check does not catch because no two rows
        # are similar enough on their own.
        if len(accepted_grams) >= 15:
            threshold = max(3, int(0.08 * len(accepted_grams)))
            hot = [g for g in grams if gram_df.get(g, 0) >= threshold]
            if len(hot) >= 2:
                reject(f"stock phrasing: {len(hot)} reused n-grams, e.g. \"{hot[0][:50]}\"")
                n_rejected += 1
                continue

        accepted_grams.append(grams)
        gram_df.update(grams)
        append_jsonl(out_path, {"id": row_id, "source_id": None, "char": char,
                                "user": "", "turns": turns,
                                "anchor_overlap": anchor_overlap})
        n_ok += 1
        print(f"row {row_id}: OK ({char}, overlap {anchor_overlap}, {len(turns)} turns)")

    print(f"\nDone: {n_ok} accepted, {n_rejected} rejected, {n_refusals} refusals -> {out_path}")
    if n_rejected > n_ok:
        print("\nMore rejected than accepted. Look at the reasons in synth_rejected.jsonl:")
        print("  'copies the anchors' in bulk  -> the model is rewriting the source; "
              "sharpen synth.system in the language pack, or lower synth.anchor_max_chars")
        print("  'duplicate' in bulk           -> the situation matrix is too small; "
              "add entries to synth.situations")
        print("  'stock phrasing' in bulk      -> raise synth.temperature, or add "
              "behaviour modifiers")


if __name__ == "__main__":
    main()
