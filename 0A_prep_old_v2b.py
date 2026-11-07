import os
import re
import sys
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ----------------- PIPELINE FORMAT PATTERNS -----------------

# Matches scene breaks (e.g. * * *, ---, ~~~)
SCENE_BREAK_RE = re.compile(r"^([\*\-\~_#]\s*){3,}$")

# Matches standard chapter/section headings under 60 chars
CHAPTER_HEADING_RE = re.compile(
    r"^(chapter\s+\d+|part\s+[a-z0-9]+|prologue|epilogue|act\s+[ivxlcdm]+|\d+|[ivxlcdm]+)$",
    re.IGNORECASE
)

# Headings to discard completely (front/back matter)
DISCARD_HEADINGS_RE = re.compile(
    r"^(table of contents|contents|copyright|about the author|acknowledgements|acknowledgments)$",
    re.IGNORECASE
)


def format_markdown_for_pipeline(markdown_text: str) -> str:
    """
    Takes Docling's structured Markdown output and formats it to the exact
    pipeline structure:
      - Strips image tags and picture captions
      - Normalizes Markdown headings to clean standalone lines (< 60 chars)
      - Ensures paragraphs are separated strictly by a blank line (\n\n)
      - De-hyphenates line wraps and cleans footnote markers
    """
    raw_blocks = re.split(r"\n\s*\n", markdown_text)
    clean_paragraphs = []

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # 1. Skip Docling image placeholders, HTML comments, and standalone images
        if block.startswith("<!--") or block.startswith("!["):
            continue

        # 2. Handle Markdown Headings (# Header, ## Header, etc.)
        heading_match = re.match(r"^#{1,6}\s*(.+)$", block)
        if heading_match:
            heading_text = heading_match.group(1).strip()

            # Discard Front/Back matter sections if they appear as headings
            if DISCARD_HEADINGS_RE.match(heading_text):
                continue

            # Keep headings alone on their line (truncated/guarded to < 60 chars)
            if len(heading_text) <= 60:
                clean_paragraphs.append(heading_text)
            else:
                # If a heading was too long, treat it as a standard paragraph
                clean_paragraphs.append(heading_text)
            continue

        # 3. Check for Scene Breaks (* * *, ---)
        if SCENE_BREAK_RE.match(block):
            clean_paragraphs.append(block)
            continue

        # 4. Process regular body paragraph:
        # Join lines within the paragraph (un-wrap hard wraps)
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        
        merged = ""
        for line in lines:
            if not merged:
                merged = line
            elif merged.endswith("-") and re.search(r"[A-Za-z]-$", merged):
                # Re-join hyphenated words (e.g., "conspir-" + "ators" -> "conspirators")
                merged = merged[:-1] + line
            else:
                merged = merged + " " + line

        # Remove footnote citations (e.g. [1], [12])
        merged = re.sub(r"\[\d+\]", "", merged)
        # Clean up Markdown bold/italic markup from text body (*italic*, **bold**)
        ## merged = re.sub(r"\*\*([^*]+)\*\*", r"\1", merged)
        ## merged = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", merged)
        # Collapse multiple spaces
        merged = re.sub(r"\s+", " ", merged).strip()

        # Discard stray page numbers that survived
        if re.match(r"^\d+$", merged):
            continue

        if merged:
            clean_paragraphs.append(merged)

    # Output strictly separated by a single blank line
    return "\n\n".join(clean_paragraphs)


def convert_pdf(pdf_path: str, output_path: str):
    print(f"Loading Docling layout engine for: {pdf_path}")

    # Configure Docling pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable local OCR for scanned pages / illustrations
    pipeline_options.do_table_structure = False  # Faster processing for text/books

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print("Analyzing document structure, columns, and reading order...")
    result = converter.convert(pdf_path)

    print("Exporting structured document...")
    docling_md = result.document.export_to_markdown()

    print("Formatting text to training pipeline specifications...")
    final_text = format_markdown_for_pipeline(docling_md)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\nDone! Pipeline-ready file saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        src = sys.argv[1]
        dest = sys.argv[2]
    else:
        src = "input/source.pdf"
        dest = "input/source.txt"

    convert_pdf(src, dest)