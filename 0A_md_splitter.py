import base64
import os
import re
import sys
from pathlib import Path

# Matches standard markdown images containing embedded base64 data:
# ![Alt Text](data:image/<ext>;base64,<data>)
IMAGE_DATA_REGEX = re.compile(
    r'!\[(?P<alt>.*?)\]\(data:image/(?P<ext>[a-zA-Z0-9\-\+]+);base64,(?P<b64>[a-zA-Z0-9+/= \r\n]+)\)'
)


def process_markdown(file_path: str):
    md_path = Path(file_path)
    if not md_path.is_file():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Output directory relative to the markdown file
    output_dir = md_path.parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    counter = 1

    def replacer(match):
        nonlocal counter
        alt_text = match.group("alt")
        ext = match.group("ext").lower()

        # Normalize common extensions
        if ext == "jpeg":
            ext = "jpg"
        elif "svg" in ext:
            ext = "svg"

        b64_data = match.group("b64").replace(" ", "").replace("\r", "").replace("\n", "")

        img_filename = f"{md_path.stem}_img_{counter}.{ext}"
        img_filepath = output_dir / img_filename

        try:
            # Decode and save the actual image
            img_bytes = base64.b64decode(b64_data)
            with open(img_filepath, "wb") as img_file:
                img_file.write(img_bytes)

            counter += 1
            # Return cleaned-up markdown link with relative path
            return f"![{alt_text}](images/{img_filename})"

        except Exception as e:
            print(f"Warning: Failed to decode image {counter}: {e}")
            return match.group(0)

    # Replace inline data with relative references
    cleaned_content = IMAGE_DATA_REGEX.sub(replacer, content)

    # Save to clean file: filename.clean.md (or overwrite if preferred)
    clean_md_path = md_path.parent / f"{md_path.stem}.clean.md"
    with open(clean_md_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    print(f"Processed: {md_path.name}")
    print(f"Extracted {counter - 1} images to: {output_dir}")
    print(f"Clean document written to: {clean_md_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_images.py <path_to_markdown_file>")
        sys.exit(1)

    process_markdown(sys.argv[1])