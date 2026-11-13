# -*- coding: utf-8 -*-
"""Interactive launcher. Run this if you do not want to memorise ten commands.

    python menu.py
    python menu.py --config config.ru.yaml
    python menu.py --no-color

It prints a numbered menu, shows which steps have already produced output, and
runs the one you pick. Nothing here does any work itself — every entry shells out
to the same script you would have typed by hand, with the same --config. So the
menu can be ignored entirely, and anything you do through it can be redone,
resumed or debugged from the command line.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import load_config, resolve_input_path, work_path  # noqa: E402
from pipeline.lang import load_pack  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

import time
import urllib.request
import urllib.error

class Palette:
    """ANSI colours, switched off when the terminal cannot show them."""

    CODES = {
        "green": "\033[38;5;82m",
        "bright_green": "\033[1;32m",
        "yellow": "\033[38;5;220m",
        "red": "\033[38;5;196m",
        "cyan": "\033[38;5;51m",
        "blue": "\033[38;5;39m",
        "magenta": "\033[38;5;201m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        "off": "\033[0m",
    }

    def __init__(self, enabled=True):
        if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
            enabled = False
        if enabled and os.name == "nt":
            os.system("")
        self.enabled = enabled

    def __call__(self, text, *styles):
        if not self.enabled:
            return text
        prefix = "".join(self.CODES[s] for s in styles if s in self.CODES)
        return f"{prefix}{text}{self.CODES['off']}" if prefix else text


ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def visible_len(text: str) -> int:
    """Return character count excluding invisible ANSI escape sequences."""
    return len(ANSI_RE.sub("", text))

def pad_visible(text: str, width: int, align: str = "left") -> str:
    """Pad string to a target width based strictly on visible glyphs."""
    vlen = visible_len(text)
    pad = max(0, width - vlen)
    if align == "right":
        return (" " * pad) + text
    elif align == "center":
        left = pad // 2
        right = pad - left
        return (" " * left) + text + (" " * right)
    return text + (" " * pad)


def check_endpoint_health(base_url, timeout=1.5):
    """Ping OpenAI-compatible server models endpoint to verify connection."""
    url = f"{base_url.rstrip('/')}/v1/models"
    start_t = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BookToDatasetMenu"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = round((time.perf_counter() - start_t) * 1000)
            if resp.status in (200, 204):
                return True, f"{elapsed_ms}ms"
            return True, f"HTTP {resp.status}"
    except Exception:
        # Some servers don't serve /v1/models; try root or return false
        return False, "offline"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def count_lines(path):
    """Number of non-empty lines, without parsing the JSON."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def is_imported(cfg):
    """True when rp_clean.jsonl came from tools/import_transcripts.py.

    Imports enter the pipeline at step 06, so steps 01-05 do not apply to that
    workdir and the menu says so instead of showing them as unfinished.
    """
    path = work_path(cfg, "rp_clean.jsonl")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        return '"imported": true' in (f.readline() or "").lower()


# Status codes, in the order of severity the menu renders them with.
DONE, PARTIAL, TODO, SKIP, INFO = "done", "partial", "todo", "skip", "info"


def check_step_debug_checkpoint(cfg, number):
    """Check if Step 01 has active atomic checkpoints in debug/."""
    debug_dir = work_path(cfg, "debug")
    if not os.path.exists(debug_dir):
        return None
    if number == "01":
        ckpts = [f for f in os.listdir(debug_dir) if f.startswith("seg_checkpoint_block_") and f.endswith(".json")]
        if ckpts:
            return f"{len(ckpts)} block ckpt(s) found"
    return None


def step_status(cfg, number):
    """Return (state, note) for one step, from what is on disk."""
    imported = is_imported(cfg)

    if number == "0A":
        src_pdf = cfg.get("input", {}).get("pdf_path", "input/source.pdf")
        txt_out = cfg["input"]["path"]
        if os.path.exists(txt_out) and os.path.getsize(txt_out) > 0:
            return DONE, f"text ready ({os.path.basename(txt_out)})"
        if os.path.exists(src_pdf):
            return TODO, f"PDF found ({os.path.basename(src_pdf)})"
        # Auto-detect any .pdf in input/
        input_dir = os.path.dirname(txt_out) or "input"
        if os.path.exists(input_dir):
            found_pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
            if found_pdfs:
                return TODO, f"PDF found ({found_pdfs[0]})"
        return TODO, "no source PDF"

    if number == "0B":
        return INFO, "diagnostic, run any time"

    if number in ("01", "02", "03", "04", "05") and imported:
        return SKIP, "not used: transcripts were imported"

    if number == "01":
        n = count_lines(work_path(cfg, "scenes.jsonl"))
        return (DONE, f"{n} scenes") if n else (TODO, "")

    if number == "02":
        total = count_lines(work_path(cfg, "scenes.jsonl"))
        n = count_lines(work_path(cfg, "scenes_attr.jsonl"))
        if not n:
            return TODO, ""
        if total and n < total:
            return PARTIAL, f"{n} of {total} scenes — rerun to continue"
        return DONE, f"{n} scenes"

    if number == "03":
        n = count_lines(work_path(cfg, "rp_raw.jsonl"))
        return (DONE, f"{n} transcripts") if n else (TODO, "")

    if number == "04":
        ok = count_lines(work_path(cfg, "rp_clean.jsonl"))
        bad = count_lines(work_path(cfg, "rp_rejected.jsonl"))
        raw = count_lines(work_path(cfg, "rp_raw.jsonl"))
        if not ok and not bad:
            return TODO, ""
        if raw and ok + bad < raw:
            return PARTIAL, f"{ok + bad} of {raw} checked — rerun to continue"
        return DONE, f"{ok} accepted, {bad} rejected"

    if number == "05":
        if not cfg["synth"]["enabled"]:
            return SKIP, "synth.enabled is false"
        n = count_lines(work_path(cfg, "synth_clean.jsonl"))
        want = cfg["synth"]["count"]
        if not n:
            return TODO, f"target {want}"
        return (PARTIAL, f"{n} of {want} — rerun to continue") if n < want else (DONE, f"{n} sessions")

    if number == "06":
        n = count_lines(cfg["format"]["output"])
        return (DONE, f"{n} sessions") if n else (TODO, "")

    if number == "07":
        if not cfg["augment"]["enabled"]:
            return SKIP, "augment.enabled is false"
        n = count_lines(cfg["augment"]["output"])
        return (DONE, f"{n} sessions") if n else (TODO, "")

    if number == "08":
        n = count_lines(cfg["assemble"]["output"])
        return (DONE, f"{n} sessions") if n else (TODO, "")

    if number == "09":
        folder = os.path.dirname(os.path.abspath(cfg["assemble"]["output"]))
        card = cfg["publish"]["card_output"]
        if not os.path.isabs(card) and os.sep not in card:
            card = os.path.join(folder, card)
        split = count_lines(os.path.join(folder, "dataset_train.jsonl"))
        if os.path.exists(card) and split:
            return DONE, f"card written, {split} train sessions"
        if os.path.exists(card):
            return PARTIAL, "card written, not split"
        return TODO, ""

    if number == "10":
        if not cfg.get("antislop", {}).get("enabled"):
            return SKIP, "antislop.enabled is false"
        n = count_lines(work_path(cfg, cfg.get("antislop", {}).get("output", "dataset_antislop.jsonl")))
        return (DONE, f"{n} sessions") if n else (TODO, "")

    return TODO, ""


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

STEPS = [
    ("0A", "0A_prep.bat", "Convert PDF to TXT"),
    ("0B", "0B_inspect.py", "Check the source text"),
    ("01", "01_segment.py", "Cut into scenes"),
    ("02", "02_attribute.py", "Find who speaks"),
    ("03", "03_convert.py", "Scenes into transcripts"),
    ("04", "04_validate.py", "Filter the transcripts"),
    ("05", "05_synthesize.py", "Generate new sessions"),
    ("06", "06_format.py", "Build the dataset"),
    ("07", "07_augment.py", "Vary the format"),
    ("08", "08_assemble.py", "Assemble the final file"),
    ("09", "09_publish.py", "Check and upload"),
    ("10", "10_antislop.py", "AntiSlop final polish"),
]

EXTRAS = [
    ("A", None, "Run steps 0A-10 unattended"),
    ("S", "tools/status.py", "Detailed progress report"),
    ("M", "tools/metrics_watch.py", "Live AntiSlop metrics"),
    ("V", None, "Read some finished sessions"),
    ("I", "tools/import_transcripts.py", "Import an already-dialogue archive"),
    ("T", None, "Offline smoke test, no GPU"),
    ("H", None, "Where to put files, how this works"),
    ("Q", None, "Quit"),
]

MARKS = {
    DONE: ("DONE", ("bright_green", "bold")),
    PARTIAL: ("PART", ("yellow", "bold")),
    TODO: ("TODO", ("dim",)),
    SKIP: ("SKIP", ("dim",)),
    INFO: ("DIAG", ("cyan",)),
}

def render_progress_bar(done_count, total_count, width=22):
    pct = done_count / max(1, total_count)
    filled = int(round(width * pct))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {int(pct * 100)}% ({done_count}/{total_count})"

HELP_TEXT = """
WHAT THIS IS
  A pipeline that turns a book, or any prose with dialogue in it, into a
  roleplay training dataset in ShareGPT format, ready for a LoRA fine-tune.

WHERE TO PUT YOUR FILES
  1. Your text as one plain UTF-8 .txt file, in input/. Blank line between
     paragraphs, chapter headings alone on their line, no page numbers or
     headers left over from a PDF.
  2. Open config.yaml and set input.path to that file, and llm.base_url to
     your model server.
  3. Everything the pipeline produces goes into the workdir from the config.
     Nothing is written next to your source text.

HOW TO USE THE MENU
  Steps are meant to be run in order, top to bottom. A green [done] means the
  step already produced output; a yellow [part] means it stopped partway and
  running it again continues from there rather than starting over.
  To redo a step from scratch, delete its output file and run it again.

IF SOMETHING LOOKS WRONG
  Read BEFORE_YOU_RUN.md. It says plainly where this works, where it works
  badly, and where it is the wrong tool for your data.
"""


def clear():
    print("\033[2J\033[H", end="") if sys.stdout.isatty() else print("\n" * 3)


def run(command, colour):
    """Run a child process and wait, so its output scrolls live."""
    print(colour("\n" + "=" * 70, "cyan"))
    print(colour("  " + " ".join(command[1:]), "cyan", "bold"))
    print(colour("=" * 70 + "\n", "cyan"))
    try:
        code = subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        print(colour("\n\ninterrupted — progress up to the last item is saved", "yellow"))
        code = 130
    print(colour("\n" + "=" * 70, "cyan"))
    if code == 0:
        print(colour("  finished", "green", "bold"))
    else:
        print(colour(f"  exited with code {code}", "red", "bold"))
    print(colour("=" * 70, "cyan"))
    input("\npress Enter to return to the menu ")


def render(cfg, pack, config_path, colour):
    """Draw the full terminal TUI dashboard with pixel-perfect alignment."""
    clear()
    W = 72  # Total inner box width

    def box_line(content_str: str) -> str:
        return colour("│", "cyan") + pad_visible(content_str, W) + colour("│", "cyan")

    # 1. Header Card
    print(colour("┌" + "─" * W + "┐", "cyan"))
    print(box_line("  BOOK-TO-DATASET PIPELINE (v5 EXPERIMENTAL)"))
    print(box_line(colour("  prose & non-fiction  -->  ShareGPT SFT dataset for LoRA", "dim")))
    print(colour("├" + "─" * W + "┤", "cyan"))

    # 2. Environment Information
    source = resolve_input_path(cfg)
    source_found = os.path.exists(source)
    source_name = os.path.basename(source) if source_found else "source.txt"
    source_status = colour("[FOUND]", "green", "bold") if source_found else colour("[MISSING]", "red", "bold")
    source_field = f"{source_name} {source_status}"

    base_url = cfg["llm"]["base_url"]
    is_live, latency = check_endpoint_health(base_url)
    model_status = colour(f"ONLINE ({latency})", "green", "bold") if is_live else colour("OFFLINE", "red", "bold")

    lang_name = pack.get("meta.name", pack.lang)
    lang_display = f"{pack.lang} ({lang_name})"

    line1 = f"  Config:   {pad_visible(config_path, 20)} Model Server: {base_url}"
    line2 = f"  Language: {pad_visible(lang_display, 20)} Status:       {model_status}"
    line3 = f"  Source:   {pad_visible(source_field, 20)} Workdir:      {cfg['workdir']}"

    print(box_line(line1))
    print(box_line(line2))
    print(box_line(line3))

    # Style note
    note = pack.get("meta.style_note")
    if note:
        first_line = str(note).strip().split("\n")[0][:W - 14]
        print(box_line(colour(f"  Note:     {first_line}", "yellow")))

    # 3. Overall Pipeline Progress Bar
    pipeline_steps = [s for s, _, _ in STEPS if s not in ("0A", "0B", "10")]
    completed_steps = sum(1 for s in pipeline_steps if step_status(cfg, s)[0] == DONE)
    progress_bar = render_progress_bar(completed_steps, len(pipeline_steps), width=20)
    print(box_line(f"  Progress: {progress_bar}"))

    # 4. Table Header
    # Column budget: Step (6) | Status (10) | Description (27) | Progress (23) + borders = 72
    print(colour("├──────┬──────────┬───────────────────────────┬────────────────────────┤", "cyan"))
    hdr = f" STEP │  STATUS  │ DESCRIPTION               │ PROGRESS / OUTPUT"
    print(colour("│", "cyan") + pad_visible(hdr, W) + colour("│", "cyan"))
    print(colour("├──────┼──────────┼───────────────────────────┼────────────────────────┤", "cyan"))

    # 5. Table Rows
    for number, _script, label in STEPS:
        state, note_text = step_status(cfg, number)
        mark, styles = MARKS[state]

        ckpt_note = check_step_debug_checkpoint(cfg, number)
        if ckpt_note and state != DONE:
            note_text = f"{note_text} ({ckpt_note})" if note_text else ckpt_note

        step_col = pad_visible(f"  {number}", 6)
        badge = colour(f"[{mark}]", *styles)
        badge_col = pad_visible(f" {badge} ", 10)
        desc_col = pad_visible(label[:25], 25)
        
        note_styled = colour(note_text[:22], "green" if state == DONE else ("yellow" if state == PARTIAL else "dim"))
        prog_col = pad_visible(note_styled, 22)

        row_str = f"{step_col}│{badge_col}│ {desc_col} │ {prog_col}"
        print(colour("│", "cyan") + pad_visible(row_str, W) + colour("│", "cyan"))

    # 6. Action Menu
    print(colour("├──────┴──────────┴───────────────────────────┴────────────────────────┤", "cyan"))
    print(box_line(colour(" ACTIONS & UTILITIES:", "bold")))

    for i in range(0, len(EXTRAS), 2):
        pair = EXTRAS[i:i + 2]
        line_items = []
        for key, _script, label in pair:
            k = colour(f"[{key}]", "cyan", "bold")
            line_items.append(pad_visible(f"  {k} {label}", 35))
        print(box_line("".join(line_items)))

    print(colour("└" + "─" * W + "┘", "cyan"))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    colour = Palette(enabled=not args.no_color)
    config_path = args.config
    cfg = load_config(config_path)
    pack = load_pack(cfg["lang"])

    by_key = {number: script for number, script, _label in STEPS}
    extras_by_key = {key: script for key, script, _label in EXTRAS}
    python = sys.executable

    while True:
        # Reloaded every pass, so editing the config in another window is
        # picked up without restarting the menu.
        try:
            cfg = load_config(config_path)
            pack = load_pack(cfg["lang"])
        except SystemExit as exc:
            print(colour(f"\nconfig problem: {exc}", "red"))
            input("fix it and press Enter ")
            continue

        render(cfg, pack, config_path, colour)
        choice = input("  choose a step, or a letter:  ").strip().upper()

        if choice in ("Q", "QUIT", "EXIT"):
            print()
            return
        if choice == "H":
            clear()
            print(HELP_TEXT)
            input("press Enter to return to the menu ")
            continue
        if choice == "A":
            run([python, "tools/run_all.py", "--config", config_path], colour)
            continue
        if choice == "T":
            clear()
            print(colour(
                "\n  The smoke test runs the whole pipeline against a fake model server,\n"
                "  in seconds, with no GPU. It checks that nothing is broken — not that\n"
                "  the output is any good.\n\n"
                "  It uses its own config and its own workdir, so your data is untouched.\n",
                "dim"))
            if input("  run it? [y/N] ").strip().lower().startswith("y"):
                run([python, "tests/run_all_smoke.sh"], colour) if os.name != "nt" else \
                    run(["bash", "tests/run_all_smoke.sh"], colour)
            continue
        if choice == "V":
            target = cfg["assemble"]["output"]
            if not os.path.exists(target):
                target = cfg["augment"]["output"]
            if not os.path.exists(target):
                target = cfg["format"]["output"]
            if not os.path.exists(target):
                print(colour("\n  No dataset built yet — run step 06 first.", "yellow"))
                input("\npress Enter ")
                continue
            run([python, "tools/inspect_dataset.py", "--sample", "5", target], colour)
            continue
        if choice == "I":
            clear()
            print(colour(
                "\n  Import is for archives that are ALREADY dialogue: chat logs, exported\n"
                "  sessions, a ShareGPT file. It replaces steps 01-05.\n\n"
                "  Give it one or more files, separated by spaces.\n", "dim"))
            raw = input("  files (blank to cancel):  ").strip()
            if not raw:
                continue
            run([python, extras_by_key["I"], *raw.split(), "--config", config_path], colour)
            continue

        # A pipeline step. Accept "0", "0A", "0B", "3", "03".
        if choice in ("0", "00"):
            key = "0A"
        elif choice.isdigit():
            key = choice.zfill(2)
        else:
            key = choice

        if key in by_key:
            state, _note = step_status(cfg, key)
            if state == SKIP:
                print(colour(f"\n  Step {key} does not apply to this workdir.", "yellow"))
                input("\npress Enter ")
                continue
            if state == DONE:
                print(colour(f"\n  Step {key} already has output. Running it again "
                             f"continues or rebuilds it; it will not duplicate work.",
                             "yellow"))
                if not input("  run anyway? [y/N] ").strip().lower().startswith("y"):
                    continue
            command = [python, by_key[key], "--config", config_path]
            if key == "09":
                if input("  upload for real, or dry run? [y = upload / N = dry run] "
                         ).strip().lower().startswith("y"):
                    pass
                else:
                    command.append("--dry-run")
            run(command, colour)
            continue
        if choice in extras_by_key and extras_by_key[choice]:
            run([python, extras_by_key[choice], "--config", config_path], colour)
            continue

        print(colour("  not a valid choice", "red"))
        input("  press Enter ")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
