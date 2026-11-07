# -*- coding: utf-8 -*-
"""Step 0A — Pluggable High-Quality PDF-to-Text Ingestion.

Hot-swap between modern deep-learning engines by changing the BACKEND variable:
  - "marker"      : datalab-to/marker (Best for books, strips headers, layout-aware)
  - "surya"       : datalab-to/surya (High-accuracy multilingual layout OCR)
  - "mineru"      : opendatalab/MinerU (Magic-PDF, exceptional on messy scans)
  - "vlm_ollama"  : Local VLM via Ollama (e.g. minicpm-v, qwen2-vl, glm-4v)
  - "native"      : Fast fallback to PyMuPDF digital text (no deep OCR)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import pymupdf as fitz

# ===========================================================================
# ⚙️ HOT-SWAP CONFIGURATION: Set your active backend here
# ===========================================================================
# Options: "marker" | "surya" | "mineru" | "vlm_ollama" | "native"
BACKEND = "marker"

# If using "vlm_ollama":
OLLAMA_MODEL = "minicpm-v"  # or "qwen2-vl", "llama3.2-vision"
OLLAMA_ENDPOINT = "http://localhost:5001"
# ===========================================================================

SCENE_BREAK_RE = re.compile(r"^([\*\-\~_#]\s*){3,}$")
DISCARD_HEADINGS_RE = re.compile(
    r"^(table of contents|contents|copyright|about the author|acknowledgements|acknowledgments)$",
    re.IGNORECASE
)


def format_markdown_for_pipeline(markdown_text: str) -> str:
    """Standardizes markdown output to submaroon pipeline specifications."""
    raw_blocks = re.split(r"\n\s*\n", markdown_text)
    clean_paragraphs = []

    for block in raw_blocks:
        b = block.strip()
        if not b or b.startswith("<!--") or b.startswith("![") or b.startswith("<picture"):
            continue

        heading_match = re.match(r"^#{1,6}\s*(.+)$", b)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            if DISCARD_HEADINGS_RE.match(heading_text):
                continue
            clean_paragraphs.append(heading_text if len(heading_text) <= 100 else heading_text)
            continue

        if SCENE_BREAK_RE.match(b):
            clean_paragraphs.append(b)
            continue

        lines = [line.strip() for line in b.splitlines() if line.strip()]
        merged = ""
        for line in lines:
            if not merged:
                merged = line
            elif merged.endswith("-") and re.search(r"[A-Za-z]-$", merged):
                merged = merged[:-1] + line
            else:
                merged = merged + " " + line

        merged = re.sub(r"\[\d+\]", "", merged)
        merged = re.sub(r"\*\*([^*]+)\*\*", r"\1", merged)
        merged = re.sub(r"\s+", " ", merged).strip()

        if re.match(r"^\d+$", merged):
            continue

        if merged:
            clean_paragraphs.append(merged)

    return "\n\n".join(clean_paragraphs)


# ---------------------------------------------------------------------------
# BACKEND 1: Marker (datalab-to/marker)
# Install: pip uninstall -y marker marker-pdf && pip install marker-pdf
# ---------------------------------------------------------------------------
def run_marker(pdf_path: str, output_path: str):
    print("Executing Marker extraction pipeline (GPU/Layout accelerated)...")
    raw_md = None

    # Method A: Try native Python API (Fastest & avoids CLI version mismatches)
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        print("  [Marker] Loaded native Python engine. Processing document...", flush=True)
        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )
        rendered = converter(pdf_path)
        raw_md = rendered.markdown
    except Exception as api_err:
        # Method B: Fallback to CLI execution if internal API changed
        print(f"  [Marker] Direct API bypass ({api_err}), trying CLI runner...", flush=True)
        with tempfile.TemporaryDirectory() as tmp_out:
            # Check for marker_single.exe or python -m marker.scripts.convert_single
            cli_cmd = shutil.which("marker_single")
            if cli_cmd:
                cmd = [cli_cmd, pdf_path, tmp_out]
            else:
                cmd = [sys.executable, "-m", "marker.scripts.convert_single", pdf_path, tmp_out]

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Marker stderr: {res.stderr}", file=sys.stderr)
                raise RuntimeError(f"Marker exited with code {res.returncode}. Ensure you ran: pip uninstall -y marker && pip install marker-pdf")

            found = [os.path.join(r, f) for r, _, fs in os.walk(tmp_out) for f in fs if f.endswith(".md")]
            if not found:
                raise FileNotFoundError("Marker completed but produced no markdown output.")

            with open(found[0], "r", encoding="utf-8") as f:
                raw_md = f.read()

    if not raw_md:
        raise RuntimeError("Marker produced empty text.")

    formatted = format_markdown_for_pipeline(raw_md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted)


# ---------------------------------------------------------------------------
# BACKEND 2: MinerU / Magic-PDF (opendatalab/MinerU)
# Install: pip install magic-pdf[full]
# ---------------------------------------------------------------------------
def run_mineru(pdf_path: str, output_path: str):
    print("Executing MinerU (Magic-PDF) extraction...")
    with tempfile.TemporaryDirectory() as tmp_out:
        cmd = ["magic-pdf", "-p", pdf_path, "-o", tmp_out, "-m", "ocr"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"MinerU stderr: {res.stderr}", file=sys.stderr)
            raise RuntimeError(f"MinerU failed with code {res.returncode}")

        found = [os.path.join(r, f) for r, _, fs in os.walk(tmp_out) for f in fs if f.endswith(".md")]
        if not found:
            raise FileNotFoundError("MinerU completed but produced no markdown output.")

        with open(found[0], "r", encoding="utf-8") as f:
            raw_md = f.read()

        formatted = format_markdown_for_pipeline(raw_md)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted)


# ---------------------------------------------------------------------------
# BACKEND 3: Surya OCR (datalab-to/surya)
# Install: pip install surya-ocr
# ---------------------------------------------------------------------------
def run_surya(pdf_path: str, output_path: str):
    print("Executing Surya OCR layout & text recognition...")
    from PIL import Image
    from surya.ocr import run_ocr
    from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    doc = fitz.open(pdf_path)
    extracted_text = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        predictions = run_ocr([img], [["en"]], det_model, det_processor, rec_model, rec_processor)
        page_lines = [line.text for line in predictions[0].text_lines]
        extracted_text.append("\n".join(page_lines))

    doc.close()
    full_text = format_markdown_for_pipeline("\n\n".join(extracted_text))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)


# ---------------------------------------------------------------------------
# BACKEND 4: Local Vision-Language Model via Ollama
# Prerequisite: ollama run minicpm-v
# ---------------------------------------------------------------------------
def run_vlm_ollama(pdf_path: str, output_path: str):
    import base64
    import requests

    print(f"Executing Local VLM via Ollama (Model: {OLLAMA_MODEL})...")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    accumulated = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        prompt = (
            "Transcribe all text on this page accurately. Maintain paragraphs. "
            "Ignore marginalia, page numbers, and decorative headers. Output only the transcribed text."
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        }

        try:
            r = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=300)
            if r.status_code == 200:
                accumulated.append(r.json().get("response", ""))
            else:
                print(f"  [warn] Page {page_idx + 1} VLM call failed: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [warn] Page {page_idx + 1} VLM error: {e}")

        if (page_idx + 1) % 10 == 0:
            print(f"  [VLM] Transcribed {page_idx + 1}/{total_pages} pages...", flush=True)

    doc.close()
    formatted = format_markdown_for_pipeline("\n\n".join(accumulated))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted)


# ---------------------------------------------------------------------------
# BACKEND 5: Native Digital Text Only (Fast baseline)
# ---------------------------------------------------------------------------
def run_native(pdf_path: str, output_path: str):
    print("Extracting native digital text blocks via PyMuPDF...")
    doc = fitz.open(pdf_path)
    accumulated = []

    for page in doc:
        blocks = page.get_text("blocks")
        page_lines = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
        if page_lines:
            accumulated.append("\n".join(page_lines))

    doc.close()
    formatted = format_markdown_for_pipeline("\n\n".join(accumulated))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted)


# ---------------------------------------------------------------------------
# Driver / CLI Entry
# ---------------------------------------------------------------------------
def dispatch_conversion(pdf_path: str, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\n[Step 0A] Converting: {pdf_path} -> {output_path}")
    print(f"[Step 0A] Active Ingestion Engine: {BACKEND.upper()}")

    if BACKEND == "marker":
        run_marker(pdf_path, output_path)
    elif BACKEND == "mineru":
        run_mineru(pdf_path, output_path)
    elif BACKEND == "surya":
        run_surya(pdf_path, output_path)
    elif BACKEND == "vlm_ollama":
        run_vlm_ollama(pdf_path, output_path)
    elif BACKEND == "native":
        run_native(pdf_path, output_path)
    else:
        sys.exit(f"Unknown BACKEND: {BACKEND}. Choose from: marker, mineru, surya, vlm_ollama, native")

    size_kb = os.path.getsize(output_path) / 1024 if os.path.exists(output_path) else 0
    print(f"[Step 0A] Success: {output_path} generated ({size_kb:.1f} KB)\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 0A: Pluggable PDF Ingestion.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--src", default=None)
    parser.add_argument("--dest", default=None)
    args, _ = parser.parse_known_args()

    dest = args.dest
    src = args.src
    if os.path.exists(args.config):
        try:
            from pipeline.config import load_config
            cfg = load_config(args.config)
            dest = dest or cfg.get("input", {}).get("path", "input/source.txt")
            src = src or cfg.get("input", {}).get("pdf_path")
        except Exception:
            pass

    dest = dest or "input/source.txt"

    if not src or not os.path.exists(src):
        input_dir = os.path.dirname(dest) or "input"
        if os.path.exists(input_dir):
            candidates = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
            if candidates:
                src = candidates[0]

    if not src or not os.path.exists(src):
        sys.exit("[0A_prep] Error: No PDF found in input/ directory.")

    dispatch_conversion(src, dest)


if __name__ == "__main__":
    main()