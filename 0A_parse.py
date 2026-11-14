# -*- coding: utf-8 -*-
"""Step 0A (Parser) — Interactive & Batch Lorebook/Forum Source Parser.

Splits oddly formatted sources (old forums, OCR'd PDFs, lorebooks) by `# `
chapter headers into sections, strips board metadata, normalizes dialogue,
converts ASCII emoticons to Unicode emoticons, and provides a Tkinter
before/after diff GUI with:
  - Color highlighting (Red = removed content, Yellow = speakers/changes,
    Green = exact/validated prose & dialogue)
  - A checkbox (enabled by default) that monitors manual edits and learns
    either:
      * LLM Few-Shot Hotswap Templates (sends before >>> after <<< chunks
        to your local LLM at llm.base_url / http://localhost:5001)
      * LLM-Synthesized Multi-Regex Rules (fast, zero-latency batch pass)
  - A Hotswap Belt (Ctrl+1 .. Ctrl+9) to save, name, apply, and erase
    reusable templates stored alongside the source file.

    python 0A_parse.py [--config config.yaml] [--all | --gui]
"""
import difflib
import json
import os
import re
import shutil
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

from pipeline.config import load_config, parse_args, resolve_input_path
from pipeline.files import read_text
from pipeline.llm import LLM, extract_json

# ---------------------------------------------------------------------------
# Emoticon & Built-in Cleanup Rules
# ---------------------------------------------------------------------------

EMOTICON_MAP = [
    (r"(?<!\S):-?\)\)(?!\S)", "😄"),
    (r"(?<!\S):-?\)(?!\S)", "😊"),
    (r"(?<!\S):-?\((?!\S)", "😞"),
    (r"(?<!\S):'\((?!\S)", "😢"),
    (r"(?<!\S):-?[dD](?!\S)", "😄"),
    (r"(?<!\S);-?\)(?!\S)", "😉"),
    (r"(?<!\S):-?[pP](?!\S)", "😛"),
    (r"(?<!\S):-?[oO](?!\S)", "😮"),
    (r"(?<!\S):-?[/\\](?!\S)", "😕"),
    (r"(?<!\S)[xX][dD](?!\S)", "😆"),
    (r"(?<!\S)<3(?!\S)", "❤️"),
    (r"(?<!\S):</3(?!\S)", "💔"),
    (r"(?<!\S):-?\*(?!\S)", "😘"),
    (r"(?<!\S)B-?\)(?!\S)", "😎"),
    (r"(?<!\S)>:-\((?!\S)", "😠"),
    (r"(?<!\S)T_T(?!\S)", "😭"),
    (r"(?<!\S)\^_\^(?!\S)", "😊"),
]

FORUM_META_PATTERNS = [
    (r"^(?:Post\s*#?\d+|#\d+)\s*$", ""),
    (r"^(?:Posted|Date|Joined|Location|Posts|Re):\s+.*$", ""),
    (r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:,?\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?\s*$", ""),
    (r"^_{3,}\s*$", ""),
    (r"^-{3,}\s*$", "---"),
    (r"^(?:Last edited by|Quote:|Originally Posted by).*$", ""),
]

SPEAKER_LINE_RE = re.compile(r"^([A-Z][A-Za-z0-9_'\- ]{1,30}):\s*(.+)$")


def convert_emoticons(text: str) -> str:
    """Replace common ASCII emoticon strings with Unicode emoticons."""
    for pattern, emoji in EMOTICON_MAP:
        text = re.sub(pattern, emoji, text)
    return text


def auto_parse_section(text: str, do_emoticons: bool = True, do_meta: bool = True) -> str:
    """Apply built-in forum metadata stripping, dialogue normalization, and emoticons."""
    out = text
    if do_meta:
        lines = out.splitlines()
        cleaned_lines = []
        for ln in lines:
            stripped = ln.strip()
            skip = False
            for pat, repl in FORUM_META_PATTERNS:
                if re.match(pat, stripped, re.IGNORECASE):
                    if repl == "":
                        skip = True
                    else:
                        ln = repl
                    break
            if not skip:
                cleaned_lines.append(ln)
        out = "\n".join(cleaned_lines)

    norm_lines = []
    for ln in out.splitlines():
        m = SPEAKER_LINE_RE.match(ln.strip())
        if m:
            speaker, speech = m.group(1).strip(), m.group(2).strip()
            if not (speech.startswith('"') or speech.startswith("“")):
                speech = f'"{speech}"'
            norm_lines.append(f"{speaker}:\n{speech}")
        else:
            norm_lines.append(ln)
    out = "\n".join(norm_lines)

    if do_emoticons:
        out = convert_emoticons(out)

    return re.sub(r"\n{3,}", "\n\n", out).strip()


def split_by_headers(text: str) -> list[dict]:
    """Split source text by `# ` chapter headers into sections."""
    lines = text.splitlines()
    sections = []
    current_header = "(Preamble)"
    current_lines = []

    for ln in lines:
        if re.match(r"^#\s+\S", ln):
            if current_lines or sections:
                sections.append({
                    "header": current_header,
                    "original": "\n".join(current_lines).strip(),
                })
            current_header = ln.strip()
            current_lines = [ln]
        else:
            current_lines.append(ln)

    if current_lines:
        sections.append({
            "header": current_header,
            "original": "\n".join(current_lines).strip(),
        })

    return [s for s in sections if s["original"]]


# ---------------------------------------------------------------------------
# Smart Hotswap Templates (LLM Few-Shot + LLM-Synthesized Regex List)
# ---------------------------------------------------------------------------

def hotswap_index_path(source_path: str) -> str:
    return f"{source_path}.hotswap.json"


def load_hotswap_templates(source_path: str) -> list[dict]:
    path = hotswap_index_path(source_path)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_hotswap_templates(source_path: str, templates: list[dict]):
    path = hotswap_index_path(source_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)


def llm_synthesize_regex_rules(llm: LLM, before: str, after: str) -> list[dict]:
    """Ask the local LLM to infer a list of regex rules from before >>> after <<<."""
    system = (
        "You are a Python regex expert. Given a BEFORE and AFTER text example, "
        "return ONLY a JSON object with a 'rules' list of {pattern, replacement} objects "
        "that transform BEFORE into AFTER (removing forum metadata, signatures, post counts, "
        "fixing line breaks, etc.)."
    )
    user = f"BEFORE:\n{before[:3500]}\n>>>\nAFTER:\n{after[:3500]}\n<<<\nReturn JSON: {{\"rules\": [{{\"pattern\": \"...\", \"replacement\": \"...\"}}]}}"
    raw = llm.chat(system, user, temperature=0.0, json_mode=True)
    data = extract_json(raw)
    rules = data.get("rules", []) if isinstance(data, dict) else []
    return [r for r in rules if isinstance(r, dict) and "pattern" in r]


def apply_hotswap_template(text: str, tpl: dict, llm: LLM | None = None) -> str:
    """Apply either a regex/regex_list template or an LLM few-shot template."""
    kind = tpl.get("type", "regex")

    if kind == "regex":
        return re.sub(tpl["pattern"], tpl.get("replacement", ""), text, flags=re.MULTILINE)

    if kind == "regex_list":
        out = text
        for r in tpl.get("rules", []):
            try:
                out = re.sub(r["pattern"], r.get("replacement", ""), out, flags=re.MULTILINE)
            except re.error:
                continue
        return out

    if kind == "llm":
        if llm is None:
            raise RuntimeError("LLM client not available for few-shot template.")
        system = (
            "You are a precise text-cleaning and formatting engine. Apply the exact same "
            "transformations shown in the EXAMPLE (fixing typos, line breaks, author attribution, "
            "removing post counts and forum signatures) to the TARGET TEXT. "
            "Preserve all actual story/lore content. Output ONLY the cleaned text, no commentary."
        )
        user = (
            f"EXAMPLE BEFORE:\n{tpl['before'][:2500]}\n>>>\n"
            f"EXAMPLE AFTER:\n{tpl['after'][:2500]}\n<<<\n\n"
            f"TARGET TEXT TO CLEAN:\n{text}"
        )
        return llm.chat(system, user, temperature=0.0).strip()

    return text


# ---------------------------------------------------------------------------
# Tkinter Interactive Diff & Hotswap GUI
# ---------------------------------------------------------------------------

class LorebookParserGUI:
    def __init__(self, root: tk.Tk, sections: list[dict], source_path: str, out_path: str, cfg: dict):
        self.root = root
        self.sections = sections
        self.source_path = source_path
        self.out_path = out_path
        self.cfg = cfg
        self.llm = LLM(cfg)
        self.idx = 0

        for sec in self.sections:
            sec.setdefault("edited", auto_parse_section(sec["original"]))
            sec.setdefault("approved", False)

        self.templates = load_hotswap_templates(self.source_path)

        self.root.title(f"0A_parse — Lorebook & Forum Parser ({os.path.basename(source_path)})")
        self.root.geometry("1280x860")

        self.learn_var = tk.BooleanVar(value=True)
        self.emoticon_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._bind_shortcuts()
        self.load_section(0)

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#eceff1", pady=6, padx=10)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="◀ Previous", command=self.prev_section).pack(side=tk.LEFT, padx=3)
        tk.Button(top, text="Next ▶", command=self.next_section).pack(side=tk.LEFT, padx=3)

        self.sec_label = tk.Label(top, text="Section 1 / 1", font=("Arial", 10, "bold"), bg="#eceff1")
        self.sec_label.pack(side=tk.LEFT, padx=12)

        tk.Checkbutton(
            top, text="Monitor manual edits & learn rules",
            variable=self.learn_var, bg="#eceff1", font=("Arial", 9, "bold")
        ).pack(side=tk.LEFT, padx=10)

        tk.Checkbutton(
            top, text="Convert Emoticons",
            variable=self.emoticon_var, bg="#eceff1", font=("Arial", 9),
            command=self.reparse_current
        ).pack(side=tk.LEFT, padx=6)

        self.status_lbl = tk.Label(top, text=f"LLM: {self.llm.base_url}", fg="#455a64", bg="#eceff1", font=("Arial", 8))
        self.status_lbl.pack(side=tk.LEFT, padx=10)

        tk.Button(
            top, text="💾 Save All & Exit", bg="#2e7d32", fg="white",
            font=("Arial", 9, "bold"), command=self.save_and_exit
        ).pack(side=tk.RIGHT, padx=4)

        # Hotswap Belt
        belt_frame = tk.LabelFrame(self.root, text=" Hotswap Belt (Ctrl+1 .. Ctrl+9) ", padx=8, pady=4)
        belt_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(4, 0))

        self.belt_buttons_frame = tk.Frame(belt_frame)
        self.belt_buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(belt_frame, text="🤖 Save as LLM Few-Shot", bg="#6a1b9a", fg="white",
                  command=lambda: self.save_current_as_template("llm")).pack(side=tk.RIGHT, padx=3)
        tk.Button(belt_frame, text="⚡ Save as LLM Regexes", bg="#1976d2", fg="white",
                  command=lambda: self.save_current_as_template("regex_list")).pack(side=tk.RIGHT, padx=3)
        tk.Button(belt_frame, text="🗑 Erase Rule", bg="#d32f2f", fg="white",
                  command=self.erase_template).pack(side=tk.RIGHT, padx=3)

        self._refresh_belt()

        # Split Diff Panes
        panes = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left_frame = tk.LabelFrame(panes, text=" BEFORE (Original — Red: Removed, Yellow: Speaker/Changed, Green: Exact) ")
        right_frame = tk.LabelFrame(panes, text=" AFTER (Editable Preview — Stays in Edit Mode) ")
        panes.add(left_frame, stretch="always")
        panes.add(right_frame, stretch="always")

        self.left_text = tk.Text(left_frame, wrap="word", font=("Consolas", 10), bg="#fafafa")
        self.left_text.pack(fill=tk.BOTH, expand=True)
        self.left_text.tag_config("removed", background="#ffcdd2", foreground="#b71c1c")
        self.left_text.tag_config("speaker", background="#fff9c4", foreground="#f57f17")
        self.left_text.tag_config("exact", background="#e8f5e9", foreground="#1b5e20")

        self.right_text = tk.Text(right_frame, wrap="word", font=("Consolas", 10), undo=True)
        self.right_text.pack(fill=tk.BOTH, expand=True)
        self.right_text.bind("<KeyRelease>", lambda _: self._on_right_edit())

        # Bottom Action Bar
        bottom = tk.Frame(self.root, pady=8, padx=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(bottom, text="✔ Approve & Next", bg="#388e3c", fg="white",
                  font=("Arial", 10, "bold"), padx=14, command=self.approve_and_next).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="✖ Deny / Revert to Original", bg="#e53935", fg="white",
                  font=("Arial", 10), padx=10, command=self.revert_current).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="🔄 Re-run Auto-Parse", bg="#f57c00", fg="white",
                  font=("Arial", 10), padx=10, command=self.reparse_current).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="🔍 Refresh Diff", command=self.update_diff_highlights).pack(side=tk.RIGHT, padx=4)

    def _bind_shortcuts(self):
        for i in range(1, 10):
            self.root.bind(f"<Control-Key-{i}>", lambda _e, idx=i - 1: self.apply_template(idx))

    def _refresh_belt(self):
        for w in self.belt_buttons_frame.winfo_children():
            w.destroy()
        if not self.templates:
            tk.Label(self.belt_buttons_frame, text="No hotswap templates saved yet. Edit text & click 'Save as LLM Few-Shot' or 'Save as LLM Regexes'.",
                     fg="#666").pack(side=tk.LEFT)
            return
        for i, tpl in enumerate(self.templates[:9]):
            icon = "🤖" if tpl.get("type") == "llm" else "⚡"
            btn = tk.Button(
                self.belt_buttons_frame,
                text=f"[Ctrl+{i + 1}] {icon} {tpl['name']}",
                command=lambda idx=i: self.apply_template(idx),
                bg="#ede7f6" if tpl.get("type") == "llm" else "#e3f2fd",
            )
            btn.pack(side=tk.LEFT, padx=3, pady=2)

    def load_section(self, idx: int):
        if not (0 <= idx < len(self.sections)):
            return
        self.idx = idx
        sec = self.sections[idx]
        status = "✔" if sec["approved"] else "○"
        self.sec_label.config(text=f"Section {idx + 1} / {len(self.sections)} {status} — {sec['header'][:45]}")

        self.left_text.config(state=tk.NORMAL)
        self.left_text.delete("1.0", tk.END)
        self.left_text.insert("1.0", sec["original"])
        self.left_text.config(state=tk.DISABLED)

        self.right_text.delete("1.0", tk.END)
        self.right_text.insert("1.0", sec["edited"])
        self.update_diff_highlights()

    def _on_right_edit(self):
        self.sections[self.idx]["edited"] = self.right_text.get("1.0", tk.END).rstrip("\n")
        self.update_diff_highlights()

    def update_diff_highlights(self):
        orig_lines = self.sections[self.idx]["original"].splitlines()
        edit_lines = self.right_text.get("1.0", tk.END).rstrip("\n").splitlines()

        self.left_text.config(state=tk.NORMAL)
        for tag in ("removed", "speaker", "exact"):
            self.left_text.tag_remove(tag, "1.0", tk.END)

        sm = difflib.SequenceMatcher(None, orig_lines, edit_lines)
        for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
            for line_idx in range(i1, i2):
                start = f"{line_idx + 1}.0"
                end = f"{line_idx + 1}.end"
                line_str = orig_lines[line_idx].strip()
                if tag == "equal":
                    if SPEAKER_LINE_RE.match(line_str):
                        self.left_text.tag_add("speaker", start, end)
                    else:
                        self.left_text.tag_add("exact", start, end)
                elif tag in ("delete", "replace"):
                    self.left_text.tag_add("removed", start, end)
        self.left_text.config(state=tk.DISABLED)

    def apply_template(self, tpl_idx: int):
        if tpl_idx >= len(self.templates):
            return
        tpl = self.templates[tpl_idx]
        current = self.right_text.get("1.0", tk.END).rstrip("\n")

        if tpl.get("type") == "llm":
            self.status_lbl.config(text=f"⏳ Applying LLM template '{tpl['name']}'...", fg="#e65100")
            self.root.update_idletasks()

            def _run():
                try:
                    updated = apply_hotswap_template(current, tpl, self.llm)
                    self.root.after(0, lambda: self._finish_apply(updated, f"LLM: {self.llm.base_url}"))
                except Exception as exc:
                    self.root.after(0, lambda: messagebox.showerror("LLM Error", str(exc)))
                    self.root.after(0, lambda: self.status_lbl.config(text=f"LLM: {self.llm.base_url}", fg="#455a64"))

            threading.Thread(target=_run, daemon=True).start()
        else:
            updated = apply_hotswap_template(current, tpl, None)
            self._finish_apply(updated, f"LLM: {self.llm.base_url}")

    def _finish_apply(self, updated: str, status_msg: str):
        self.right_text.delete("1.0", tk.END)
        self.right_text.insert("1.0", updated)
        self.sections[self.idx]["edited"] = updated
        self.status_lbl.config(text=status_msg, fg="#455a64")
        self.update_diff_highlights()

    def save_current_as_template(self, mode: str):
        if not self.learn_var.get():
            messagebox.showinfo("Monitoring Off", "Enable 'Monitor manual edits & learn rules' first.")
            return
        before = self.sections[self.idx]["original"]
        after = self.right_text.get("1.0", tk.END).rstrip("\n")
        if before.strip() == after.strip():
            messagebox.showwarning("No Changes", "Edit the right pane before saving a rule.")
            return

        name = simpledialog.askstring("Save Hotswap Rule", "Enter a name for this Hotswap Template:", parent=self.root)
        if not name:
            return

        if mode == "llm":
            tpl = {"name": name.strip(), "type": "llm", "before": before, "after": after}
            self.templates.append(tpl)
            save_hotswap_templates(self.source_path, self.templates)
            self._refresh_belt()
        else:
            self.status_lbl.config(text="⏳ Synthesizing regexes via LLM...", fg="#e65100")
            self.root.update_idletasks()
            try:
                rules = llm_synthesize_regex_rules(self.llm, before, after)
                if not rules:
                    messagebox.showwarning("No Rules", "LLM did not return any regex rules.")
                    return
                tpl = {"name": name.strip(), "type": "regex_list", "rules": rules}
                self.templates.append(tpl)
                save_hotswap_templates(self.source_path, self.templates)
                self._refresh_belt()
                messagebox.showinfo("Saved", f"Synthesized {len(rules)} regex rules into '{name.strip()}'.")
            except Exception as exc:
                messagebox.showerror("LLM Error", f"Failed to synthesize regexes:\n{exc}")
            finally:
                self.status_lbl.config(text=f"LLM: {self.llm.base_url}", fg="#455a64")

    def erase_template(self):
        if not self.templates:
            return
        names = [f"{i + 1}: {t['name']} ({t.get('type', 'regex')})" for i, t in enumerate(self.templates)]
        choice = simpledialog.askinteger(
            "Erase Hotswap Rule",
            "Enter rule number to erase:\n" + "\n".join(names),
            minvalue=1, maxvalue=len(self.templates), parent=self.root
        )
        if choice:
            self.templates.pop(choice - 1)
            save_hotswap_templates(self.source_path, self.templates)
            self._refresh_belt()

    def prev_section(self):
        self.sections[self.idx]["edited"] = self.right_text.get("1.0", tk.END).rstrip("\n")
        if self.idx > 0:
            self.load_section(self.idx - 1)

    def next_section(self):
        self.sections[self.idx]["edited"] = self.right_text.get("1.0", tk.END).rstrip("\n")
        if self.idx + 1 < len(self.sections):
            self.load_section(self.idx + 1)

    def approve_and_next(self):
        self.sections[self.idx]["edited"] = self.right_text.get("1.0", tk.END).rstrip("\n")
        self.sections[self.idx]["approved"] = True
        if self.idx + 1 < len(self.sections):
            self.load_section(self.idx + 1)
        else:
            messagebox.showinfo("Done", "All sections reviewed! Click 'Save All & Exit' to write output.")

    def revert_current(self):
        orig = self.sections[self.idx]["original"]
        self.sections[self.idx]["edited"] = orig
        self.right_text.delete("1.0", tk.END)
        self.right_text.insert("1.0", orig)
        self.update_diff_highlights()

    def reparse_current(self):
        orig = self.sections[self.idx]["original"]
        parsed = auto_parse_section(orig, do_emoticons=self.emoticon_var.get())
        self.sections[self.idx]["edited"] = parsed
        self.right_text.delete("1.0", tk.END)
        self.right_text.insert("1.0", parsed)
        self.update_diff_highlights()

    def save_and_exit(self):
        self.sections[self.idx]["edited"] = self.right_text.get("1.0", tk.END).rstrip("\n")
        backup = f"{self.out_path}.bak"
        if os.path.exists(self.out_path) and not os.path.exists(backup):
            shutil.copy2(self.out_path, backup)
        final_text = "\n\n".join(s["edited"] for s in self.sections if s["edited"].strip())
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(final_text + "\n")
        messagebox.showinfo("Saved", f"Saved {len(self.sections)} sections to:\n{self.out_path}")
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    def extra(parser):
        parser.add_argument("--all", action="store_true", help="A) Parse all entries at once without GUI")
        parser.add_argument("--gui", action="store_true", help="B) Parse one section at a time in Tkinter GUI")
        parser.add_argument("--input", default=None, help="Override input file path")
        parser.add_argument("--output", default=None, help="Override output file path (default: overwrite input)")

    args = parse_args(__doc__.splitlines()[0], extra)
    cfg = load_config(args.config)
    source_path = args.input or resolve_input_path(cfg)
    out_path = args.output or source_path

    text = read_text(source_path, cfg["input"]["encodings"])
    sections = split_by_headers(text)
    print(f"Loaded {source_path}: {len(sections)} section(s) detected by `# ` headers.")

    mode = "all" if args.all else ("gui" if args.gui else None)
    if mode is None:
        print("\nChoose parsing mode:")
        print("  [A] Parse all entries at once (batch)")
        print("  [B] Parse ONE section at a time (Tkinter diff & hotswap GUI)")
        choice = input("Select [A/B]: ").strip().upper()
        mode = "all" if choice == "A" else "gui"

    if mode == "all":
        templates = load_hotswap_templates(source_path)
        llm = LLM(cfg) if any(t.get("type") == "llm" for t in templates) else None
        parsed_sections = []
        for i, sec in enumerate(sections, 1):
            print(f"  [{i}/{len(sections)}] {sec['header'][:50]}", flush=True)
            txt = auto_parse_section(sec["original"], do_emoticons=True)
            for tpl in templates:
                txt = apply_hotswap_template(txt, tpl, llm)
            if txt.strip():
                parsed_sections.append(txt.strip())
        backup = f"{out_path}.bak"
        if os.path.exists(out_path) and not os.path.exists(backup):
            shutil.copy2(out_path, backup)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parsed_sections) + "\n")
        print(f"Batch-parsed {len(parsed_sections)} sections -> {out_path}")
    else:
        root = tk.Tk()
        LorebookParserGUI(root, sections, source_path, out_path, cfg)
        root.mainloop()


if __name__ == "__main__":
    main()