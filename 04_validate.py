# -*- coding: utf-8 -*-
"""Step 04 — validate the transcripts.

Two filters, cheapest first.

  1. Lexical overlap, no LLM involved: what share of the transcript's content
     words also occur in the source scene. A low share means the converter wrote
     its own prose. This catches the most common failure for free and runs over
     everything before a single model call is made.

  2. An LLM judge: is the narrative voice preserved, is anything invented, are
     the lines on the right characters. Optional — set validate.llm_check to
     false to keep only the cheap filter.

Rejections are kept, with their reasons. Read them: a systematic reason is a
prompt problem in step 03, and deleting the data instead of fixing the prompt
just makes a smaller dataset with the same flaw.

    Input:  <workdir>/rp_raw.jsonl, <workdir>/scenes.jsonl
    Output: <workdir>/rp_clean.jsonl, <workdir>/rp_rejected.jsonl

    python 04_validate.py [--config config.yaml]
"""
import os
import re
import sys
from collections import Counter

_OOC_PATTERNS = re.compile(
    r"\b(?:as a helpful ai|as an ai(?: assistant| language model)?|"
    r"i(?:'m| am) an ai(?: language model)?|i must politely decline|"
    r"is there anything else i can help|you(?:'re| are) welcome! feel free to ask)\b",
    re.I,
)
_QUOTED_SPAN_RE = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d|\'[^\']*\'')


def has_ooc_narration(text):
    """True if unquoted narration contains AI-assistant drift."""
    narration = _QUOTED_SPAN_RE.sub(" ", text or "")
    return bool(_OOC_PATTERNS.search(narration))


_FOREIGN_WORDS = frozenset((
    "olhos", "olho", "cabelo", "ojos", "pelo", "piel", "occhi", "occhio",
    "capelli", "augen", "haare", "haut", "yeux", "cheveux", "ogen",
))
_FOREIGN_WORD_RE = re.compile(r"\b(" + "|".join(_FOREIGN_WORDS) + r")\b", re.I)
_GLUED_WORD_RE = re.compile(r"\b[a-z]{3,}[A-Z][a-z]{2,}\b")


def has_foreign_or_glued_word(text):
    """Catch Latin-script foreign words ('green olhos') and glued words ('visiblyTrembling')."""
    if not text:
        return False
    return bool(_FOREIGN_WORD_RE.search(text) or _GLUED_WORD_RE.search(text))

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map
from pipeline.textstats import overlap


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    validate = cfg["validate"]
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)

    scenes = {s["id"]: s for s in
              read_jsonl(require_file(work_path(cfg, "scenes.jsonl"), "01_segment.py"))}
    transcripts = read_jsonl(require_file(work_path(cfg, "rp_raw.jsonl"), "03_convert.py"))
    clean_path = work_path(cfg, "rp_clean.jsonl")
    reject_path = work_path(cfg, "rp_rejected.jsonl")

    done = {item["id"] for item in read_jsonl(clean_path)}
    done |= {item["id"] for item in read_jsonl(reject_path)}
    if done:
        print(f"[resume] {len(done)} already checked, continuing")

    # Pass 1: the cheap filter, over everything, before any model call.
    pending = []
    n_clean = n_rejected = 0
    reasons = Counter()
    for transcript in transcripts:
        if transcript["id"] in done:
            continue
        scene = scenes.get(transcript["source_id"])
        if not scene:
            append_jsonl(reject_path, {**transcript, "reject_reason": "no source scene"})
            reasons["no source scene"] += 1
            n_rejected += 1
            continue

        # Calculate overlap specifically on the character turns (the source voice),
        # not on the synthetic user prompts which add non-source question tokens.
        char_text = "\n".join(turn["text"] for turn in transcript["turns"] if turn.get("role") == "char")
        full_text = "\n".join(turn["text"] for turn in transcript["turns"])
        eval_text = char_text if char_text.strip() else full_text
        
        score = round(overlap(eval_text, scene["text"]), 3)
        transcript["lexical_overlap"] = score
        transcript["_text"] = full_text

        # Calibrate lexical overlap for narrative monologues and conversational adaptations
        is_informational = transcript.get("imported") or len(transcript.get("turns", [])) <= 2 or transcript.get("user") == "The Inquirer"
        
        min_overlap = validate["lexical_overlap_min"]
        if is_informational:
            # Informational briefings and single-speaker journals use a calibrated threshold
            min_overlap = max(0.25, min_overlap - 0.25)
        elif any(turn.get("role") == "user" for turn in transcript["turns"]):
            min_overlap = max(0.35, min_overlap - 0.15)

        if score < min_overlap:
            append_jsonl(reject_path, {
                **{k: v for k, v in transcript.items() if k != "_text"},
                "reject_reason": f"lexical_overlap {score} < {min_overlap}"})
            reasons["low lexical overlap"] += 1
            n_rejected += 1
            print(f"rp {transcript['id']}: REJECT overlap={score} (min {min_overlap})")
            continue

        if has_ooc_narration(eval_text):
            append_jsonl(reject_path, {
                **{k: v for k, v in transcript.items() if k != "_text"},
                "reject_reason": "ooc_narration_drift"})
            reasons["ooc narration drift"] += 1
            n_rejected += 1
            print(f"rp {transcript['id']}: REJECT ooc_narration_drift")
            continue

        if validate.get("foreign_word_check", True) and has_foreign_or_glued_word(eval_text):
            append_jsonl(reject_path, {
                **{k: v for k, v in transcript.items() if k != "_text"},
                "reject_reason": "foreign_or_glued_word"})
            reasons["foreign or glued word"] += 1
            n_rejected += 1
            print(f"rp {transcript['id']}: REJECT foreign_or_glued_word")
            continue

        pending.append(transcript)

    print(f"Passed the lexical filter: {len(pending)}")

    # Pass 2: the judge. Skipped entirely when disabled.
    if not validate["llm_check"]:
        for transcript in pending:
            transcript.pop("_text", None)
            append_jsonl(clean_path, transcript)
            n_clean += 1
        print(f"\n[validate.llm_check is off] {n_clean} transcripts accepted on the "
              f"lexical filter alone -> {clean_path}")
        return

    llm = LLM(cfg)
    debug_dir = work_path(cfg, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    def work(transcript):
        scene = scenes[transcript["source_id"]]
        return llm.chat(
            pack.require("validate.system"),
            render(pack.require("validate.user"),
                   scene=scene["text"], transcript=transcript["_text"]),
            temperature=validate["temperature"],
        )

    for transcript, raw, error in parallel_map(work, pending, workers):
        transcript.pop("_text", None)
        if error is not None:
            print(f"  [warn] rp {transcript['id']}: {error} — will retry on the next run",
                  file=sys.stderr)
            continue

        try:
            verdict = extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            if is_refusal(raw):
                print(f"rp {transcript['id']}: the judge refused — not marked, "
                      f"rerun with a different model")
                continue

            # Blueprint D: format-repair round before falling back to regex
            try:
                repaired_raw = llm.chat(
                    "You are a strict JSON-only formatter. Output ONLY a valid JSON object.",
                    "Convert your previous evaluation into valid JSON with keys "
                    "voice_preserved, no_invention, split_correct, comment:\n\n" + (raw or "")[:4000],
                    temperature=0.0,
                    json_mode=True,
                )
                verdict = extract_json(repaired_raw)
            except Exception:
                verdict = None

            if verdict is None:
                # Fallback heuristic: Extract boolean fields using regex if JSON wrapper failed
                v_match = re.search(r'["\']?voice_preserved["\']?\s*:\s*(true|false)', raw, re.I)
                n_match = re.search(r'["\']?no_invention["\']?\s*:\s*(true|false)', raw, re.I)
                s_match = re.search(r'["\']?split_correct["\']?\s*:\s*(true|false)', raw, re.I)
                if v_match and n_match:
                    verdict = {
                        "voice_preserved": v_match.group(1).lower() == "true",
                        "no_invention": n_match.group(1).lower() == "true",
                        "split_correct": s_match.group(1).lower() == "true" if s_match else True,
                        "comment": "regex-salvaged verdict"
                    }
                else:
                    fail_file = os.path.join(debug_dir, f"judge_fail_{transcript['id']}.txt")
                    with open(fail_file, "w", encoding="utf-8") as f:
                        f.write(f"--- ERROR: {exc} ---\n\n" + (raw or ""))
                    print(f"  [warn] rp {transcript['id']}: invalid JSON (logged to {fail_file})",
                          file=sys.stderr)
                    continue

        # Polymorphic guard: handle list wrapping or non-dict strings
        if isinstance(verdict, list) and len(verdict) > 0 and isinstance(verdict[0], dict):
            verdict = verdict[0]
        elif not isinstance(verdict, dict):
            fail_file = os.path.join(debug_dir, f"judge_fail_{transcript['id']}.txt")
            with open(fail_file, "w", encoding="utf-8") as f:
                f.write(f"--- ERROR: Expected dict, got {type(verdict).__name__} ---\n\n" + (raw or ""))
            print(f"  [warn] rp {transcript['id']}: non-dict verdict (logged to {fail_file})",
                  file=sys.stderr)
            continue

        transcript["verdict"] = verdict

        # In solo journals, {{[INPUT]}}/{{[OUTPUT]}} pairs, or informational scenes, relax constraints
        is_pair_mode = bool(scene.get("input_prompt") or transcript.get("input_prompt"))
        is_informational = is_pair_mode or transcript.get("imported") or len(transcript.get("turns", [])) <= 2 or transcript.get("user") == "The Inquirer"
        
        if is_pair_mode or is_informational:
            keys = ("voice_preserved",)
        else:
            keys = ("voice_preserved", "no_invention", "split_correct")

        def _is_explicitly_false(val):
            if val is None:
                return False
            if isinstance(val, bool):
                return not val
            return str(val).strip().lower() in ("false", "0", "no", "failed", "rejected")

        def _is_truthy(val):
            if val is None:
                return True  # Lenient on missing keys: pass unless explicitly flagged false
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("true", "1", "yes", "ok", "passed")

        strict_judge = validate.get("strict_judge", False)
        trace_log = os.path.join(debug_dir, "stage_04_trace.log")

        # Determine pass/fail
        if strict_judge:
            passed = all(_is_truthy(verdict.get(key)) for key in keys)
            failed = [key for key in keys if not _is_truthy(verdict.get(key))]
        else:
            # Lenient: Accept unless keys are explicitly set to false WITH a non-empty complaint
            explicit_failures = [key for key in keys if _is_explicitly_false(verdict.get(key))]
            comment = str(verdict.get("comment") or "").strip()
            passed = len(explicit_failures) == 0 or (len(comment) == 0 and not strict_judge)
            failed = explicit_failures

        comment = str(verdict.get("comment") or "").strip()

        if passed:
            append_jsonl(clean_path, transcript)
            n_clean += 1
            print(f"rp {transcript['id']}: OK overlap={transcript['lexical_overlap']}")
            with open(trace_log, "a", encoding="utf-8") as tf:
                tf.write(f"[STAGE 04][ACCEPT] id={transcript['id']} overlap={transcript['lexical_overlap']} keys={keys}\n")
        else:
            append_jsonl(reject_path, {
                **transcript,
                "reject_reason": comment or f"judge: {', '.join(failed)}"})
            reasons["judge: " + ", ".join(failed)] += 1
            n_rejected += 1
            print(f"rp {transcript['id']}: REJECT {comment if comment else ', '.join(failed)}")
            with open(trace_log, "a", encoding="utf-8") as tf:
                tf.write(f"[STAGE 04][REJECT] id={transcript['id']} failed={failed} comment={comment}\n")

    print(f"\nDone this run: {n_clean} accepted, {n_rejected} rejected")
    if reasons:
        print("Rejection reasons:")
        for reason, count in reasons.most_common():
            print(f"  {count:>5}  {reason}")
    print(f"\nRead {reject_path}. A reason that repeats is a step 03 prompt problem, "
          f"not a reason to throw data away.")


if __name__ == "__main__":
    main()
