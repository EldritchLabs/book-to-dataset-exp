# -*- coding: utf-8 -*-
"""Step 06 — write the plain ShareGPT dataset.

Takes the validated transcripts, plus the synthetic rows if step 05 produced
any, and turns them into training sessions with one system card each.

This is a complete, trainable dataset on its own. Step 07 is optional and
improves instruction-following, but if you only want a style adapter you can
stop here and train on this file.

What the step enforces:
  * every session gets the system card from the language pack, with the
    character name substituted;
  * a session that would start on a character turn gets a short scene opener as
    its first human turn, picked deterministically by row id so the openers stay
    evenly distributed;
  * consecutive turns of the same role are merged, and a trailing human turn is
    dropped, because an unanswered prompt is not a training target.

    Input:  <workdir>/rp_clean.jsonl, optionally <workdir>/synth_clean.jsonl
    Output: format.output (default dataset_base.jsonl)

    python 06_format.py [--config config.yaml]
"""
import os

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import read_jsonl, require_file, write_jsonl
from pipeline.lang import load_pack, render
from pipeline.sharegpt import session_stats, to_sharegpt
from pipeline.textstats import percentile, pareto_repair_turn_text


def load_rows(cfg):
    """Converted transcripts plus synthetic rows, with non-clashing ids."""
    rows = read_jsonl(require_file(work_path(cfg, "rp_clean.jsonl"), "04_validate.py"))
    synth_path = work_path(cfg, "synth_clean.jsonl")
    if os.path.exists(synth_path):
        synth = read_jsonl(synth_path)
        # Synthetic ids restart from zero, so they are shifted past the
        # converted ones. Ids drive the deterministic opener choice, and
        # colliding ids would give two rows the same opener.
        base = max((row["id"] for row in rows), default=-1) + 1
        for offset, row in enumerate(synth):
            row["id"] = base + offset
            row.setdefault("source_id", None)
        rows += synth
        print(f"[synthetic] {len(synth)} rows added from {synth_path}")
    return rows


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])

    card_template = cfg["format"]["system_card"] or pack.require("cards.full")
    openers = pack.require("openers")
    fallback_user = pack.fallback_user

    rows = load_rows(cfg)
    assert rows, "no transcripts to format — check the output of step 04"

    dataset = []
    fallback_char = pack.fallback_char

    debug_dir = work_path(cfg, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    trace_log = os.path.join(debug_dir, "stage_06_trace.log")

    for row in rows:
        char_name = row.get("char") or fallback_char
        user_name = row.get("user") or fallback_user
        opener = render(openers[row["id"] % len(openers)],
                        char=char_name, user=user_name)
        # Apply Pareto-optimal cleanups across turns prior to packaging
        repaired_turns = []
        for t in row.get("turns", []):
            clean_txt = pareto_repair_turn_text(t.get("text", ""))
            if clean_txt:
                repaired_turns.append({"role": t["role"], "text": clean_txt})

        session = to_sharegpt(
            repaired_turns,
            render(card_template, char=char_name),
            opener=opener,
        )
        if session:
            dataset.append(session)
            with open(trace_log, "a", encoding="utf-8") as tf:
                tf.write(f"[STAGE 06][FORMAT] id={row['id']} char={char_name} turns={len(row.get('turns', []))}\n")
        else:
            with open(trace_log, "a", encoding="utf-8") as tf:
                tf.write(f"[STAGE 06][DROP] id={row['id']} char={char_name} reason=invalid_sharegpt_shape\n")

    output = cfg["format"]["output"]
    write_jsonl(output, dataset)

    stats = session_stats(dataset)
    lengths = stats["gpt_words"]
    print(f"\nDone: {stats['sessions']} sessions, {stats['messages']} messages -> {output}")
    if lengths:
        print(f"Character turn length in words: p10 {percentile(lengths, .1)}, "
              f"median {percentile(lengths, .5)}, p90 {percentile(lengths, .9)}")
    print(f"Unique system blocks: {stats['unique_system_blocks']}")
    if stats["unique_system_blocks"] <= 5:
        print("\nThat low number is expected here: one card per character. It is also "
              "the reason step 07 exists — a model that never sees the instruction "
              "change learns not to read it.")
    print("\nTrain with train_on_inputs: false, so the loss lands on the gpt turns only.")


if __name__ == "__main__":
    main()
