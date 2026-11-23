import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def load_records(filepath: str):
    """
    Loads JSON records robustly:
    1. Attempts line-by-line JSONL parsing.
    2. Attempts standard whole-file JSON parsing (array or object).
    3. Falls back to streaming/concatenated JSON decoding, ignoring trailing incomplete data.
    """
    # Strategy 1: Standard line-by-line JSONL
    line_records = []
    has_valid_line = False
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    line_records.append(obj)
                    has_valid_line = True
            except json.JSONDecodeError:
                # If a line isn't valid standalone JSON, line-by-line parsing won't suffice
                has_valid_line = False
                break

    if has_valid_line and line_records:
        return line_records

    # Strategy 2 & 3: Whole buffer decode (handles JSON arrays, multi-line JSON, or concatenated objects)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # Try standard whole-file decode first (clean arrays / objects)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: Streaming decode (extracts all complete JSON objects, safely ignoring trailing incomplete data)
    decoder = json.JSONDecoder()
    records = []
    idx = 0
    length = len(raw)

    # If wrapped in an unclosed or broken list, skip opening bracket
    while idx < length and raw[idx] in (" ", "\t", "\r", "\n", "["):
        idx += 1

    while idx < length:
        while idx < length and raw[idx] in (" ", "\t", "\r", "\n", ",", "]"):
            idx += 1
        if idx >= length:
            break
        try:
            obj, end_idx = decoder.raw_decode(raw, idx)
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict):
                records.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            # Stop if corrupted / truncated tail is reached
            break

    return records


def dump_jsonl_line(item) -> str:
    """Serializes a single JSON object safely onto one physical line."""
    return (
        json.dumps(item, ensure_ascii=False)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def convert_a_to_b(records):
    """Converts Format A (rp_clean.jsonl) -> Format B (ShareGPT)."""
    out = []
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        convs = []
        sys_prompt = (row.get("system_prompt") or "").strip()
        if sys_prompt:
            convs.append({"from": "system", "value": sys_prompt})

        for t in row.get("turns", []):
            if not isinstance(t, dict):
                continue
            role = t.get("role", "")
            text = (t.get("text") or "").strip()
            if not text:
                continue
            if role == "char":
                convs.append({"from": "gpt", "value": text})
            elif role == "user":
                convs.append({"from": "human", "value": text})

        if convs:
            out.append({
                "id": row.get("id", idx + 1),
                "conversations": convs,
            })
    return out


def convert_b_to_a(records, default_char="CHAR", default_user="USER"):
    """Converts Format B (ShareGPT) -> Format A (rp_clean.jsonl)."""
    out = []
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        turns = []
        sys_prompt = ""
        for msg in row.get("conversations", []):
            if not isinstance(msg, dict):
                continue
            frm = str(msg.get("from", "")).lower()
            val = (msg.get("value") or "").strip()
            if not val:
                continue
            if frm == "system":
                sys_prompt = val
            elif frm in ("gpt", "assistant"):
                turns.append({"role": "char", "text": val})
            elif frm in ("human", "user"):
                turns.append({"role": "user", "text": val})

        if turns:
            item = {
                "id": row.get("id", idx),
                "source_id": row.get("source_id", idx),
                "char": row.get("char", default_char),
                "user": row.get("user", default_user),
                "turns": turns,
            }
            if sys_prompt:
                item["system_prompt"] = sys_prompt
            out.append(item)
    return out


class ConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RP <-> ShareGPT Format Converter")
        self.geometry("540x320")
        self.resizable(False, False)

        self.mode_var = tk.StringVar(value="A2B")
        self.b_style_var = tk.StringVar(value="json")
        self.in_path_var = tk.StringVar()
        self.out_path_var = tk.StringVar()
        self.char_var = tk.StringVar(value="CHAR")
        self.user_var = tk.StringVar(value="USER")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        dir_frame = ttk.LabelFrame(self, text="Conversion Direction")
        dir_frame.pack(fill="x", **pad)
        ttk.Radiobutton(dir_frame, text="A (rp_clean.jsonl)  ->  B (ShareGPT)", variable=self.mode_var, value="A2B").pack(side="left", padx=12, pady=6)
        ttk.Radiobutton(dir_frame, text="B (ShareGPT)  ->  A (rp_clean.jsonl)", variable=self.mode_var, value="B2A").pack(side="left", padx=12, pady=6)

        file_frame = ttk.LabelFrame(self, text="Files")
        file_frame.pack(fill="x", **pad)

        ttk.Label(file_frame, text="Input:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(file_frame, textvariable=self.in_path_var, width=48).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(file_frame, text="Browse...", command=self._browse_in).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(file_frame, text="Output:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(file_frame, textvariable=self.out_path_var, width=48).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(file_frame, text="Browse...", command=self._browse_out).grid(row=1, column=2, padx=6, pady=4)

        opt_frame = ttk.LabelFrame(self, text="Options")
        opt_frame.pack(fill="x", **pad)

        ttk.Label(opt_frame, text="Format B Output:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(opt_frame, text="JSON Array (.json)", variable=self.b_style_var, value="json").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Radiobutton(opt_frame, text="JSONL (.jsonl)", variable=self.b_style_var, value="jsonl").grid(row=0, column=2, sticky="w", padx=4)

        ttk.Label(opt_frame, text="Default Char (B->A):").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(opt_frame, textvariable=self.char_var, width=16).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(opt_frame, text="Default User (B->A):").grid(row=1, column=2, sticky="e", padx=4)
        ttk.Entry(opt_frame, textvariable=self.user_var, width=14).grid(row=1, column=3, sticky="w", padx=4)

        ttk.Button(self, text="Convert", command=self._run_convert).pack(pady=12)

    def _browse_in(self):
        path = filedialog.askopenfilename(filetypes=[("JSON / JSONL", "*.jsonl *.json"), ("All files", "*.*")])
        if path:
            self.in_path_var.set(path)
            if not self.out_path_var.get():
                base, _ = os.path.splitext(path)
                ext = ".json" if (self.mode_var.get() == "A2B" and self.b_style_var.get() == "json") else ".jsonl"
                self.out_path_var.set(f"{base}_converted{ext}")

    def _browse_out(self):
        path = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=[("JSONL", "*.jsonl"), ("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.out_path_var.set(path)

    def _run_convert(self):
        in_path = self.in_path_var.get().strip()
        out_path = self.out_path_var.get().strip()
        if not in_path or not os.path.isfile(in_path):
            messagebox.showerror("Error", "Please select a valid input file.")
            return
        if not out_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        try:
            records = load_records(in_path)
            if not records:
                messagebox.showwarning("Warning", "No valid JSON records could be loaded from the file.")
                return

            if self.mode_var.get() == "A2B":
                converted = convert_a_to_b(records)
                with open(out_path, "w", encoding="utf-8") as f:
                    if self.b_style_var.get() == "json":
                        json.dump(converted, f, indent=2, ensure_ascii=False)
                    else:
                        for item in converted:
                            f.write(dump_jsonl_line(item) + "\n")
            else:
                converted = convert_b_to_a(
                    records,
                    self.char_var.get().strip() or "CHAR",
                    self.user_var.get().strip() or "USER"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in converted:
                        f.write(dump_jsonl_line(item) + "\n")

            messagebox.showinfo("Success", f"Converted {len(converted)} entries ->\n{out_path}")
        except Exception as e:
            messagebox.showerror("Conversion Failed", str(e))


if __name__ == "__main__":
    ConverterGUI().mainloop()