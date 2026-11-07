# book-to-dataset-exp v4

*A high-yield synthetic dataset pipeline adapted for both prose fiction and complex non-fiction literature, using LLMs to generate ShareGPT SFT datasets from books you provide, with optional Anti-Slop integration from Sam Paech.*

---

## Provenance & Credits

* **Upstream Foundation (v1):** Originally designed and authored by **SubMaroon** with assistance from **Anthropic Claude**. Built to produce the training corpus for the [Boulesis-v2.1-26B-A4B](https://huggingface.co/SubMaroon/Boulesis-v2.1-26B-A4B) model family.
* **Experimental Architecture & Refactor (v4):** Expanded, hardened, and adapted by **EldritchLabs** with assistance from **Google Gemini**. Transforms a fiction-centric extraction pipeline into a crash-resilient framework capable of processing non-fiction, historical monographs, investigative exposés, and technical works without discarding information or collapsing into dry Q&A pairs.
* **Anti-Slop:** Regenerate GPT turns via AntiSlop API. Original code by **Sam Paech**.

---

## Contents

- [What You Get](#what-you-get)
- [Why This Fork Exists: The Non-Fiction Problem](#why-this-fork-exists-the-non-fiction-problem)
- [Comprehensive Changelog (v1 vs v4)](#comprehensive-changelog-v10-vs-v40)
- [Document Ingestion Architecture (Step 0A & Auxiliary Scripts)](#document-ingestion-architecture-step-0a--auxiliary-scripts)
- [Detailed Pipeline Step Reference (0A through 10)](#detailed-pipeline-step-reference-0a-through-10)
- [The Interactive Menu](#the-interactive-menu)
- [The Legacy Menu](#the-legacy-menu)
- [Requirements & Setup](#requirements--setup)
- [Try It In Two Minutes](#try-it-in-two-minutes)
- [Preparing Source Documents](#preparing-source-documents)
- [Configuration Reference (`config.yaml`)](#configuration-reference-configyaml)
- [Language Packs & Prompt Mechanics (`prompts/en.yaml`)](#language-packs--prompt-mechanics-promptsenyaml)
- [Checkpoints, Quality Audits & Inspection](#checkpoints-quality-audits--inspection)
- [Overnight Execution & Automated Gates](#overnight-execution--automated-gates)
- [Importing Pre-Existing Dialogue](#importing-pre-existing-dialogue)
- [Multilingual Execution & Merging](#multilingual-execution--merging)
- [Adding New Languages](#adding-new-languages)
- [The Diluent: Off-Distribution Hardening](#the-diluent-off-distribution-hardening)
- [Behavioral Directives via Presets](#behavioral-directives-via-presets)
- [Publishing & Local Distribution](#publishing--local-distribution)
- [Fine-Tuning Guidance](#fine-tuning-guidance)
- [Troubleshooting & Failure Modes](#troubleshooting--failure-modes)
- [Scope and Limits](#scope-and-limits)
- [Repository Layout](#repository-layout)
- [Anchors Note](#anchors-note)

---

## What You Get

A ShareGPT-formatted JSONL file where every record represents a high-density training conversation:

```json
{"conversations": [
  {"from": "system", "value": "You are Barbara Walker and the scene's narrator..."},
  {"from": "human",  "value": "*The story continues. Barbara Walker looks up.*"},
  {"from": "gpt",    "value": "Talismans? Well, that's one way to put it. But I think it's more accurate to say that they're symbols of initiation..."}
]}
```

Four distinct session topologies populate the resulting training data:

1. **Converted Fiction Scenes:** Narrative fiction converted into alternating character/user turns. Scene narration is carried over verbatim to retain authorial voice.
2. **Dramatized Expository Scenes:** Historical, investigative, or non-fiction prose converted into diegetic, immersive discussions between a living expert character and an inquisitive, pushback-capable interlocutor.
3. **Synthetic Grounded Sessions:** Scenarios synthesized by the model that never occurred in the text, anchored to real character cognition models rather than verbatim text copying.
4. **Format-Augmented Variants:** The same core sessions rewritten under orthogonal typographic, grammatical, and structural constraints (direct speech vs. embedded prose, past vs. present tense, dense vs. airy paragraphs, inner-thought blocks).

---

## Why This Fork Exists: The Non-Fiction Problem

The original v1 pipeline was strictly designed for prose fiction with direct dialogue. When pointed at complex non-fiction literature (e.g., investigative compendiums, historical monographs, or detailed technical manuscripts), upstream pipelines fail catastrophically:

* **Massive Text Mega-Blocks:** Chapters in non-fiction books often use descriptive titles rather than `"Chapter X"`, bypassing structural detection and dumping 500,000+ characters into a single LLM sliding window.
* **Severe Data Starvation:** In non-fiction, direct speech quotation share drops below the default threshold (`min_dialogue_share: 0.03`). Upstream discarded 95% to 100% of the book as "unusable".
* **The "Turn-1 Data Dump" Defect:** Naive attempts to convert exposition into Q&A cause the model to output a 300-word encyclopedia entry on Turn 1, answering every follow-up question before it is asked, resulting in verbatim sentence looping across turns 2–6.
* **The Interrogation Loop:** The simulated user sounds like a student reading a quiz sheet, eliminating narrative tension and destroying conversational elasticity.

`book-to-dataset-exp` resolves these systemic flaws by decoupling document ingestion, adding real-time atomic disk checkpointing, and deploying a diegetic roleplay converter that treats non-fiction as a living scene.

---

## Comprehensive Changelog (v1 vs v4)

### 1. Architectural Overview

| Subsystem | Upstream v1 | Experimental v4 (This Fork) |
| :--- | :--- | :--- |
| **Document Ingestion** | Manual pre-cleaning required; only `.txt` files supported. | Automated Step `0A`: Native Docling PDF conversion + base64 image extraction + `.md`/`.txt` auto-resolution. |
| **Target Document Scope** | Fiction novels with explicit spoken dialogue. | Fiction novels **plus** expository non-fiction, historical treatises, and monographs. |
| **Segmentation Fault Tolerance** | In-memory sliding window per block. A failure on paragraph 1,800/1,871 lost all work. | Real-time atomic disk checkpointing (`os.fsync`) per window. Instant resume across crashes. |
| **Model Inference Routing** | Global `llm.base_url` for every pipeline step. | Dedicated per-step model/endpoint overrides (`llm.segment`, `llm.convert`, etc.). |
| **Format Flexibility** | `.txt` exclusively. Markdown headers ignored. | Native `.md` and `.txt` support; Markdown `#` to `####` headers treated as structural bounds. |
| **Automated Execution** | `tools/run_all.py` ran steps `00`–`08`. | Extended automation (`A` in menu) runs `0A` through `10` unattended with batch-script execution. |
| **Dataset Publishing** | Required remote Hugging Face API credentials. | Local-first: splits, validates, and builds dataset cards entirely on local disk without network calls. |

---

### 2. File-by-File Changes

#### `0A_prep.bat` & Ingestion Tooling (New in v4)
* **Replaced Unstable Python OCR Chains:** Abandoned brittle embedded OCR pipelines (EasyOCR, Tesseract C++ wrappers, and MagicPDF/Detectron2 which fail on Windows due to missing DLLs and uncompiled C++ dependencies).
* **Native CLI Ingestion:** Executes `docling input/sample.pdf` directly via batch processing, converting 1,000+ page image/text PDFs into clean, structured Markdown in a single pass.
* **`0A_md_splitter.py`:** Parses embedded base64 image strings (`data:image/...;base64,...`), decodes them into discrete binary image files in an adjacent `/images` directory, and rewrites the source into clean, relative Markdown links (`![Alt](images/sample_img_1.png)`). This preserves the text structure for SFT while assembling image assets for future Vision-Language Model (VLM) training.

#### `menu.py`
* **Step 0A & 0B Integration:** Integrated document preparation (`0A_prep.bat`) and inspection (`0B_inspect.py`) directly into the status board.
* **Input Path Auto-Detection:** Automatically inspects `input/` for `.pdf`, `.md`, or `.txt` files rather than hardcoding static filenames.
* **Batch Shell Dispatch:** Added native `cmd.exe /c` execution hooks for `.bat` and `.cmd` steps, preventing Python from attempting to import shell scripts as Python modules.
* **Dynamic Menu Routing:** Fixed single-digit key normalization so typing `0`, `0a`, or `0A` launches the ingestion stage rather than falling through to Step 0B or Step 03.

#### `tools/run_all.py`
* **Extended Automation Range:** Updated the automated suite default range from `--from 01 --to 08` to `--from 0A --to 10`.
* **Subprocess Execution Gate:** Implemented explicit handling for `.bat` files via `subprocess.call(..., shell=True)` so batch-driven ingestion executes smoothly inside unattended runs.

#### `pipeline/config.py`
* **Dual Format Resolution (`resolve_input_path`):** Replaced static file reading with an intelligent resolver. If `source.txt` is missing, it automatically checks for `source.md`. If neither exists, it scans `input/` for any valid Markdown or text document.
* **Expository Configuration Keys:** Added `convert.allow_informational: true` to prevent non-fiction or documentary scenes from being discarded when lacking dialogue.
* **Bite-Sized Narrative Thresholds:** Calibrated default segmentation caps (`max_scene_chars: 4000`, `max_block_chars: 5000`, `llm_window_chars: 4500`) to prevent context window saturation and token-limit truncation on 12B/14B parameter models.
* **Extended Limits:** Increased default token budgets (`max_tokens: 4096`), timeouts (`timeout: 6000`), and default anchoring budgets to handle dense academic subject matter.

#### `pipeline/llm.py`
* **Dynamic Per-Step Routing:** `LLM.__init__(cfg, step_name=None)` reads sub-configuration dictionaries matching the calling step (`cfg["llm"]["segment"]`, `cfg["llm"]["convert"]`). Allows deploying small, fast models (e.g., Llama 3.2 3B) for structural line attribution and boundary detection, while routing high-parameter reasoning models (e.g., Qwen 2.5 14B/32B) to conversion and synthesis.
* **Multi-Tier JSON Salvage Engine:** Completely overhauled `extract_json()` with heuristic string repairs:
  - Automatically closes unclosed quotation marks, arrays, and bracket stacks when generation cuts off at `max_tokens`.
  - Fixes trailing commas inside arrays/objects and ignores conversational chat preambles.
  - Salvages truncated boundary arrays (`{"boundaries": [...]}`).
  - Extracts turns using non-greedy regex patterns across unescaped quotes (`{"role": "...", "text": "..."}`).
  - Catches reversed key order (`{"text": "...", "role": "..."}`) and bracketed tags (`[char]: ... [user]: ...`).

#### `01_segment.py`
* **Real-Time Checkpointing Engine:** Implemented `seg_checkpoint_block_{id}.json` and `scenes_discovered_block_{id}.json` state persistence. Every completed sliding window flushes discovered boundaries and formatted scene previews to disk using `os.fsync`.
* **Crash-Resilient Resume:** Automatically queries disk checkpoints upon launch. A block of 2,000 paragraphs interrupted at paragraph 1,618 resumes instantly from paragraph 1,618 on the next run.
* **Markdown Heading Recognition:** Integrated `MD_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$")` alongside language pack chapter regexes. Markdown headings are treated as structural boundaries, preventing non-fiction books from coalescing into a single mega-block.
* **Polymorphic Boundary Parsing:** Handles both dictionary outputs (`{"boundaries": [...]}`) and raw list payloads (`[...]`), resolving `'list' object has no attribute 'get'` crashes.

#### `02_attribute.py`
* **Polymorphic Unpacker:** Safely extracts character and line dictionaries whether the model returns an object `{"characters": ...}` or a list `[{"characters": ...}]`.
* **Expository Document Fallback:** Gracefully assigns empty lines and characters for non-narrative document fragments, clearance sheets, and tables without failing validation.
* **Diagnostic Failure Dumps:** Dumps unparseable model responses to `work/<lang>/debug/attr_fail_<scene_id>.txt` for immediate inspection.

#### `03_convert.py`
* **Pre-Flight Scene Repair & Filtering (`repair_and_filter_scenes`):** Validates and cleans `scenes_attr.jsonl` before calling the model, stripping markdown image links, OCR checkbox soup (`O O`, `0 0`), and rule lines into `scenes_attr_repaired.jsonl`, while logging unrecoverable binary/table fragments to `skipped_garbage_scenes.jsonl`.
* **Diegetic Expository Conversion:** Replaced the mechanical Q&A prompt with an immersive roleplay dramatization engine. When a scene lacks spoken dialogue, it converts the factual text into a multi-turn conversation between an expert/narrator (`meta.fallback_char`) and an active, curious partner (`meta.fallback_user`).
* **Configurable Anti-Data-Dump Constraint:** Configurable via `convert.anti_dump: "on" | "off" | "random"`. Rather than hard-coding strict turn pacing, users can toggle strict incremental disclosure (`on`), allow unconstrained comprehensive answers (`off`), or use a reproducible Bernoulli trial (`random`) to balance turn lengths across the corpus. If set to `on`, it enforces explicit negative rules preventing the character from explaining the entire passage in Turn 1; facts are revealed incrementally across conversational turns.
* **Configurable Cinematic Action Weaving:** Configurable via `convert.cinematic_beats: "on" | "off" | "random"`. Allows fine control over whether the model strictly weaves physical mannerisms and sensory stage directions into every turn (`on`), strips them for clean verbal discussions (`off`), or deterministically alternates between the two styles (`random`). If set to `on`, it mandates that expository character turns interweave factual assertions with physical actions, environmental reactions, pauses, and expressive mannerisms.
* **Aggressive Turn-Level Salvaging:** Extracts turns from cut-off or malformed model responses via multi-tier regex patterns, guaranteeing valid roleplay dialogues even if JSON delimiters break.
* **Diagnostic Logging:** Logs unparseable replies to `work/<lang>/debug/convert_fail_<scene_id>.txt` and dropped short transcripts to `convert_dropped_<scene_id>.txt`.

#### `04_validate.py`
* **Judge Failure Dumps:** Logs unparseable judge verdicts directly to `work/<lang>/debug/judge_fail_<id>.txt` rather than raising uncaught exceptions.

#### `05_synthesize.py`
* **Resilient Character Rotation:** Replaced the hard-exit exception with an intelligent multi-tier fallback: rotates top speakers with 5+ lines, falls back to named entities present in scene metadata, and defaults to `meta.fallback_char` for non-fiction corpora.
* **Automated Missing-Turn Synthesis:** If generation yields only a single turn (an answer without a question, or a question without an answer), the engine automatically calls a targeted completion prompt to generate the complementary counterpart, preserving valid roleplay exchanges instead of discarding rows.
* **Enforced JSON Mode & Turn Salvaging:** Guarantees `json_mode=True` and applies secondary regex turn salvaging, dumping failed generations to `work/<lang>/debug/synth_fail_<row_id>.txt`.

#### `06_format.py`
* **Name Normalization & Opener Guard:** Automatically substitutes `meta.fallback_char` and `meta.fallback_user` if character attributes are missing, preventing unrendered `[[char]]` tokens and clashing session IDs.

#### `07_augment.py`
* **Incremental Disk Checkpointing & Crash Resilience:** Replaced in-memory accumulation with an atomic disk cache (`aug_accepted_partial.jsonl`). Every accepted format variant is flushed immediately, preventing the loss of hundreds of completed rewrites if a long run is interrupted.
* **Live ETA & Job Progress:** Real-time console reporting displaying completion percentages, elapsed time, and dynamically estimated completion times (ETA) per variant.
* **Turn Recovery for Rewrites:** Adds turn-level regex salvaging for format rewrites, preventing valid re-stylings with unescaped dialogue quotes from being discarded as `"llm error"`.

#### `08_assemble.py`
* **Path-Safe Source Resolution:** `resolve_source_input()` resolves paths relative to the current working directory as well as absolute paths and direct `work_path()` lookups, seamlessly finding `dataset_aug.jsonl` or `dataset_base.jsonl`.

#### `09_publish.py`
* **Fully Offline Operation:** Removed the mandatory Hugging Face Hub upload barrier. The script validates ShareGPT syntax, creates train/eval splits, writes dataset documentation cards, and saves output to local storage without requiring API tokens or remote repositories.

#### `10_antislop.py`
* **Sequential Turn & Rate Monitoring:** Enhanced the console interface with live percentage completion, assistant turn counts, elapsed processing time, and ETA estimates per session.

---

## Document Ingestion Architecture (Step 0A & Auxiliary Scripts)

PDF ingestion is handled by an external batch pipeline, keeping the Python engine clean and free of brittle C++ compilation wrappers:

```
[Raw Scanned / Text PDF]
          │
          ▼  0A_prep.bat
   [Docling Engine]
          │
          ▼  sample.md (Markdown with base64 embedded images)
   0A_md_splitter.py
     ├── Decodes base64 images ──> input/images/sample_img_X.png
     └── Rewrites Markdown    ──> input/source.md (Clean, relative image links)
```

### The Primary Toolchain (`0A_prep.bat`)
`0A_prep.bat` executes IBM's `docling` directly against the source PDF in `input/`. Docling reads the PDF using DocLayNet layout segmentation models:
1. It parses reading order across multiple columns.
2. It isolates sidebars, pull-quotes, and figure captions from the primary text.
3. It outputs clean, structural Markdown with embedded figures converted to base64 data URIs.

`0A_md_splitter.py` then scans the Markdown, decodes each base64 string into its native binary representation (`.png`, `.jpg`, `.svg`), stores it in `input/images/`, and updates the Markdown with clean relative paths.

### Auxiliary Ingestion Backups (Optional)
Two historical fallback engines are included in the repository for reference or specialized environments:
* **`0A_prep_old_v2b.py`:** Direct Docling Python API integration. Uses `PdfPipelineOptions(do_ocr=True)`. It handles standard layouts well, but can cause CPU hangs on Windows if RapidOCR encounters complex photographic collages without CUDA acceleration.
* **`0A_prep_old_v2e.py`:** A pluggable multi-backend orchestrator designed to switch between `marker-pdf`, `surya-ocr`, `magic-pdf` (MinerU), and local vision models via Ollama. *(Note: MinerU and Marker rely on Linux-specific dependencies like Meta's `detectron2` and C++ audio extensions that frequently fail on native Windows without WSL2).*

---

## Detailed Pipeline Step Reference (0A through 09)

The pipeline is organized into linear, auditable, and resumable steps:

```
[0A_prep.bat]       Extract PDF to Markdown + separate images
     │
[0B_inspect.py]     Inspect source text; verify block distributions
     │
[01_segment.py]     Segment text into narrative & thematic scenes
     │
[02_attribute.py]   Identify participants and map lines to speakers
     │
[03_convert.py]     Convert scenes into ShareGPT conversational transcripts
     │
[04_validate.py]    Filter hallucinations via lexical overlap & judge passes
     │
[05_synthesize.py]  Generate off-script sessions anchored to real character logic
     │
[06_format.py]      Package transcripts into ShareGPT JSON objects
     │
[07_augment.py]     Vary format directives across sessions
     │
[08_assemble.py]    Compile final training dataset with optional diluent
     │
[09_publish.py]     Validate schema, split train/eval, write dataset card
     │
[10_antislop.py]    (Optional) Regenerate GPT turns via AntiSlop API
```

### Note
You have to run `0A_prep.bat` on its own outside of `menu.py` for now.

### Step 0A — Ingestion (`0A_prep.bat`)
Converts input PDFs into Markdown. Rips embedded images to disk and generates `input/source.md`.

### Step 0B — Inspection (`0B_inspect.py`)
Analyzes the source text without invoking LLMs. Evaluates block length distribution, dialogue density, chapter markers, and explicit scene breaks, providing recommendations for config tuning.

### Step 01 — Segmentation (`01_segment.py`)
Splits continuous text into scenes. Structural markers (chapter titles, Markdown `#` headers, and `* * *` breaks) are cut for free. Large prose sections exceeding `segment.max_block_chars` pass through an LLM sliding window. Saves progress to `work/en/debug/seg_checkpoint_block_{id}.json` after every window. Caps scene lengths (default 4,000 chars) to prevent context exhaustion in downstream steps.

### Step 02 — Attribution (`02_attribute.py`)
Determines which characters are present in each scene and attributes spoken dialogue. Unattributed lines are flagged as `unknown`. Polymorphic parsing safely handles list or dictionary responses, while non-narrative document fragments cleanly output empty line arrays without halting. Failures are written to `work/en/debug/attr_fail_{id}.txt`.

### Step 03 — Conversion (`03_convert.py`)
Performs pre-flight data sanitization (`repair_and_filter_scenes`) to repair malformed JSON lines and eliminate non-narrative form noise into `scenes_attr_repaired.jsonl`. Then converts scenes into multi-turn dialogue transcripts:
* **Fiction Mode:** Alternating turns where the character carries environmental narration and their own dialogue, while the interlocutor provides reactive turns. Narration is carried over verbatim to retain authorial voice.
* **Expository Non-Fiction Mode (`allow_informational: true`):** Converts historical, biographical, or technical exposition into an immersive discussion between an in-universe expert (`meta.fallback_char`) and an active conversational partner (`meta.fallback_user`).
  - **Configurable Anti-Data-Dump Constraint (`convert.anti_dump`):**
    - `on`: Enforces strict incremental disclosure and action beats across turns, eliminating Turn-1 knowledge dumps.
    - `off`: Permits expansive, rich monologues on Turn 1 if the source text calls for it.
    - `random`: Uses a deterministic Bernoulli seed per scene to mix terse action beats with detailed exposition throughout the dataset.
  - **Configurable Cinematic Action Weaving (`convert.cinematic_beats`):**
    - `on`: Mandates environmental gestures, physical mannerisms, pauses, and sensory beats interwoven with speech.
    - `off`: Produces dialogue-focused turns without forced physical stage directions or descriptive padding.
    - `random`: Reproducible Bernoulli mix ensuring the training corpus contains both rich descriptive scenes and crisp, fast-paced dialogue exchanges.
* **Turn Recovery & Failure Auditing:** Employs multi-tier regex turn salvaging to recover turns across unescaped quotes or formatting glitches, writing unrecoverable replies to `work/en/debug/convert_fail_{id}.txt`.

**Alternate Description:**
Transforms scenes into multi-turn dialogue transcripts.
* **Fiction Mode**: Converts narrative prose and direct speech into alternating character/user turns. The character's lines plus the surrounding narration become one turn; the interlocutor's turns are their lines only, bare, with no narration. **That asymmetry is deliberate:** it teaches the model to answer richly to thin input, which is what a real chat session looks like. In a crowd scene, everyone else becomes part of the world; their lines go inside the character's narration.
* **Expository / Non-Fiction Mode (`allow_informational: true`)**: Converts narrative exposition into a diegetic roleplay discussion. Enforces progressive, incremental disclosures with sensory action beats, eliminating the "turn-1 knowledge dump."

### Step 04 — Transcript Validation (`04_validate.py`)
A two-tier gate:
1. **Lexical Overlap:** Measures the proportion of content words shared between the transcript and the source scene. Paraphrased summaries that drop authorial voice are rejected.
2. **LLM Judge (Optional):** Evaluates voice preservation, checks for factual inventions, and verifies line-to-speaker assignments. Rejections are logged to `work/en/rp_rejected.jsonl`, and unparseable judge replies are saved to `work/en/debug/judge_fail_{id}.txt`.

**Alternate Description:**
Guards against hallucination and structural drift.
* **Lexical Overlap Filter (Cheapest filter first):** Validates that the transcript shares a configured baseline of content words with the source scene. A low share means the converter wrote its own prose instead of carrying the source over. This runs over everything before a single model call is made.
* **Judge Pass:** Evaluates whether the narrative voice is retained, checks for factual inventions, and verifies line-to-speaker assignments. Rejections are kept with their reasons in `rp_rejected.jsonl`.

### Step 05 — Anchored Synthesis (`05_synthesize.py`)
Creates novel conversational scenarios that do not exist in the source book. Shows the model two real scenes as an anchor for *how the character reasons*. Uses a randomized situation matrix combined with character rotations. If attributed speaker lines are sparse, it automatically falls back to scene entity lists or language pack defaults. If generation yields only 1 turn (e.g. an isolated character monologue or unreciprocated user query), it automatically invokes `synthesize_missing_turn` to synthesize the missing counterpart turn into a complete conversational pair. Enforces `json_mode=True` with turn salvaging and logs unparseable generations to `work/en/debug/synth_fail_{id}.txt`.

**Alternate Description:**
Creates entirely new scenarios utilizing the source's characters and reasoning patterns. Without these, the adapter quotes the source instead of generalizing from it.
* Grounds the model using two real scenes as stylistic references shown as *a sample of how the character thinks*, not as prose to imitate.
* Filters out near-duplicates via n-gram Jaccard similarity and catches formulaic phrasing (stock skeletons reused across many rows).

### Step 06 — ShareGPT Formatting (`06_format.py`)
Compiles validated transcripts and synthetic rows into ShareGPT format. Prepends system cards from the language pack, normalizes turn orders, and inserts short, deterministic scene openers when a session opens on a character turn.

**Alternate Description:**
Standardizes validated transcripts and synthetic rows into ShareGPT JSON objects.
* Prepends system cards with dynamic character name substitution.
* Normalizes turn order, merges consecutive turns of the same role, and ensures every conversation starts on a user turn. This is already a trainable dataset; if you only want a style adapter, you can stop here.

### Step 07 — Formatting Augmentation (`07_augment.py`)
Rewrites transcripts across randomized formatting axes (e.g., straight quotes vs. embedded speech, first-person vs. third-person, dense vs. airy paragraphs, inner thoughts). Writes matching directives into the system prompt, teaching the model to attend to system instructions rather than ignoring them. Includes turn recovery parsing to salvage rewrites with unescaped internal quotes.
* **Atomic Resume & Progress Tracking:** Flushes every accepted variant to `work/<lang>/aug_accepted_partial.jsonl` on completion. If a run of 500+ jobs is stopped, restarting Step 07 resumes immediately from the exact pending job with live ETA estimations.

**Alternate Description:**
Prevents models from ignoring system instructions during inference.
* If every row carries the same system prompt, the model learns that the system block carries no information and stops attending to it. This step rewrites each transcript under diverse formatting directives (e.g., direct quotes vs. embedded prose, first-person vs. third-person, dense vs. airy paragraphs, inner thoughts).
* The rewrite receives three things at once: the source scene as the **style** reference, the finished transcript as the **content**, and the directive list as the **target format**. Anchoring style to the source scene rather than to the transcript is what keeps this from being a copy of a copy.
* **Three guards protect each variant:**
  1. **Refusal:** the model declined instead of rewriting.
  2. **Retention:** softening — distinctive words replaced by vague ones. (Checked before overlap on purpose, because softening lowers both, and reporting "low overlap" for what is actually softening sends you looking in the wrong place).
  3. **Overlap:** the rewriter substituting its own vocabulary.

### Step 08 — Assembly (`08_assemble.py`)
Combines augmented data (or Step 06 base data) with optional third-party general roleplay data (the diluent). Strips macro tags, applies safety filters, balances language shares, and shuffles the final training file.

### Step 09 — Final Validation & Publishing (`09_publish.py`)
Runs automated data-hygiene checks (ensuring no trailing human turns, no empty turns, and no malformed macros). Performs a session-level train/eval split, generates an informative dataset card (`DATASET_CARD.md`), and exports ready-to-train datasets locally.

### Step 10 — AntiSlop Final Polish (`10_antislop.py`)
A specialized optional post-processing step that connects to Sam Paech's [AntiSlop Sampler](https://github.com/sam-paech/antislop-sampler). Because AntiSlop works by backtracking logits *during generation*, this step reads your assembled dataset and **regenerates all assistant responses** through the AntiSlop API, ensuring zero AI-clichés (e.g., "testament to", "shivers down my spine") make it into the final dataset. Features sequential progress monitoring with assistant turn counts, elapsed time, and ETA calculations.

#### Patch 1: `pipeline/config.py`
Added the default configuration block for `antislop`.`

#### Patch 2: `config.yaml`
Added the user-facing config block.

#### Patch 3: `menu.py`
Added Step 10 to the menu UI and update the unattended run range.

#### Patch 4: `tools/run_all.py`
Added Step 10 to the automated runner.

**Prerequisite:** You must run the AntiSlop API locally in a separate terminal:
```bash
git clone https://github.com/sam-paech/antislop-sampler.git
cd antislop-sampler
pip install fastapi uvicorn ipywidgets IPython transformers bitsandbytes accelerate
python run_api.py --model unsloth/Llama-3.2-3B-Instruct --slop_adjustments_file slop_phrase_prob_adjustments.json
```
Enable it in `config.yaml` (`antislop.enabled: true`) and run:
```bash
python 10_antislop.py
```

---

## The Interactive Menu

Launch the terminal control interface with:

```bash
python menu.py
python menu.py --config config.yaml
```

```text
┌────────────────────────────────────────────────────────────────────────┐
│  BOOK-TO-DATASET PIPELINE (v4 EXPERIMENTAL)                            │
│  prose & non-fiction  -->  ShareGPT SFT dataset for LoRA               │
├────────────────────────────────────────────────────────────────────────┤
│  Config:   config.yaml          Model Server: http://localhost:5001    │
│  Language: en (English)         Status:       ONLINE (6ms)             │
│  Source:   source.md [FOUND]    Workdir:      work/en                  │
│  Note:     The shipped prompts aim at serious, character-driven        │
│  Progress: [========------------] 44% (4/9)                            │
├──────┬──────────┬───────────────────────────┬──────────────────────────┤
│ STEP │  STATUS  │ DESCRIPTION               │ PROGRESS / OUTPUT        │
├──────┼──────────┼───────────────────────────┼──────────────────────────┤
│  0A  │  [TODO]  │ Convert PDF to TXT        │ no source PDF            │
│  0B  │  [DIAG]  │ Check the source text     │ diagnostic, run any t    │
│  01  │  [DONE]  │ Cut into scenes           │ 29 scenes                │
│  02  │  [DONE]  │ Find who speaks           │ 29 scenes                │
│  03  │  [DONE]  │ Scenes into transcripts   │ 28 transcripts           │
│  04  │  [DONE]  │ Filter the transcripts    │ 9 accepted, 19 reject    │
│  05  │  [PART]  │ Generate new sessions     │ 13 of 200 - rerun to     │
│  06  │  [TODO]  │ Build the dataset         │                          │
│  07  │  [TODO]  │ Vary the format           │                          │
│  08  │  [TODO]  │ Assemble the final file   │                          │
│  09  │  [TODO]  │ Check and upload          │                          │
│  10  │  [TODO]  │ AntiSlop final polish     │                          │
├──────┴──────────┴───────────────────────────┴──────────────────────────┤
│ ACTIONS & UTILITIES:                                                   │
│  [A] Run steps 0A-10 unattended   [S] Detailed progress report         │
│  [V] Read some finished sessions  [I] Import an already-dialogue       │
│  [T] Offline smoke test, no GPU   [H] Where to put files, how this work│
│  [Q] Quit                                                              │
└────────────────────────────────────────────────────────────────────────┘
```

* **Live Server Status:** Pings your local LLM model server on every refresh, displaying live connectivity and millisecond latency.
* **Unified Progress Bar:** Visual progress bar calculates overall pipeline completion rate.
* **Checkpoint Awareness:** Automatically scans `work/debug/` and flags in-progress sliding windows (`ckpt found`).
* **Visual Status Badges:**
  - `[✔ DONE]`: Completed with session/scene yield counts.
  - `[◐ PART]`: Interrupted or partially complete; pressing the step resumes from the last disk flush without reprocessing.
  - `[○ TODO]`: Pending execution.
  - `[— SKIP]`: Disabled in config (e.g. diluent or antislop).
  - `[ℹ DIAG]`: Diagnostic utilities runnable at any time.
* **Action Commands:** Press `A` to run Steps 0A through 10 unattended; `S` prints the yield funnel; `V` previews random formatted sessions; `T` launches the offline mock-server test suite; `I` imports dialogue transcripts.

**Fixes:**
1. Added a helper function `strip_ansi(text)` and `ansi_pad(text_with_ansi, width)` that pads strings based solely on **visible length**.
2. Replaced unsupported Unicode glyphs with universal clean ASCII/extended-ASCII symbols (`[DONE]`, `[PART]`, `[TODO]`, `[DIAG]`, `[SKIP]`, and ASCII progress bars `[####....]`) so it renders symmetrically across all Windows and Linux consoles without font corruption.
3. Fixed the fixed-width table column arithmetic so the enclosing box border `│` lands on the exact same column for every single row.

---

## The Legacy Menu

If you prefer the original minimalist GUI then run:
```bash
python menu_legacy.py
python menu_legacy.py --config config.yaml
```

```text
  ROLEPLAY DATASET PIPELINE
  prose with dialogue  ->  ShareGPT dataset for a LoRA fine-tune

  config   config.yaml
  language en (English)   prompts/en.yaml
  source   input/source.md  [found]
  workdir  work/en
  model    http://localhost:5001

  ------------------------------------------------------------------
  [done]  0A  Convert PDF to TXT          text ready (source.md)
  [ -- ]  0B  Check the source text       diagnostic, run any time
  [done]  01  Cut into scenes             84 scenes
  [part]  02  Find who speaks             45 of 84 scenes — rerun to continue
  [    ]  03  Scenes into transcripts
  [    ]  04  Filter the transcripts
  [    ]  05  Generate new sessions       target 200
  [    ]  06  Build the dataset
  [    ]  07  Vary the format
  [    ]  08  Assemble the final file
  [    ]  09  Check and upload
  [    ]  10  AntiSlop Final Polish
  ------------------------------------------------------------------
          A   Run steps 0A-10 unattended
          S   Detailed progress report
          V   Read some finished sessions
          I   Import an already-dialogue archive
          T   Offline smoke test, no GPU
          H   Where to put files, how this works
          Q   Quit
```

* **Color Indicators:** Green (`[done]`) denotes completed steps; Yellow (`[part]`) marks partially processed steps with active resume checkpoints; Dim (`[    ]`) marks pending steps.
* **Action Commands:** Press `A` to run Steps 0A through 10 unattended; `S` prints the current yield funnel; `V` displays random finished sessions; `T` launches an offline mock-server test suite.

---

## Requirements & Setup

### Core Installation
The core pipeline relies on Python 3.10+ and standard dependencies:

```bash
pip install -r requirements.txt
pip install docling pymupdf
```

*Optional packages:*
```bash
pip install pandas pyarrow      # Required only if importing Parquet diluent datasets
pip install huggingface_hub     # Required only if uploading to Hugging Face
```

### LLM Serving Endpoints
You need an OpenAI-compatible local API server running an instruct-tuned or reasoning-tuned model:
* **vLLM:** `vllm serve Qwen/Qwen2.5-14B-Instruct --port 5001`
* **llama-server (llama.cpp):** `llama-server -m model.gguf --port 5001 -c 8192`
* **KoboldCPP:** Launch with your selected model, enabling the OpenAI API endpoint on port 5001.

---

## Try It In Two Minutes

Before pointing this at your own text, run it against the built-in fixtures. There is a mock server that speaks the OpenAI API and returns plausible output for every step, so the whole pipeline runs end to end in seconds with no GPU:

```bash
# terminal 1
python tools/mock_llm_server.py --port 8999

# terminal 2
python tools/run_all.py --config tests/config.test.yaml
python 09_publish.py --config tests/config.test.yaml --dry-run
```

That exercises the plumbing: JSON parsing, resume, the quality gates, the ShareGPT conversion, the reports. It tells you nothing about quality — the mock returns mechanical text — but it will catch a broken config or a missing dependency before you spend a night of compute. Use the same trick after any change to the config or a language pack.

---

## Preparing Source Documents

This is the step people skip, and it is the one that decides how good the result is. The pipeline segments on structure first and only falls back to the model where structure is missing, so **every marker you leave in the file saves time and improves the segmentation**.

To achieve optimal extraction, place your input files directly in `input/`:

1. **Native PDFs / Scanned PDFs:** Place your book as `input/sample.pdf`. Step 0A will automatically parse it to `input/source.md` and extract all images to `input/images/`.
2. **Text / Markdown Files:** If you already have text, save it as `input/source.md` or `input/source.txt` (UTF-8 encoded).

### Structural Markers
The pipeline leverages structural patterns to avoid unnecessary LLM calls:
* **Paragraphs:** Must be separated by a **blank line** (`\n\n`), not indentation alone.
* **Headings:** Standalone lines under 80 characters (e.g., `# Chapter 1`, `Part Two`, `Prologue`, bare Roman numerals).
* **Scene Breaks:** Three or more repeated divider characters (`* * *`, `---`, `~~~`).
* **Dialogue:** Guillemets, straight quotes, curly quotes, and dash-led lines are all recognized.

### What to remove before you start (if bypassing Step 0A):
If you are providing your own `.txt` file instead of using the `0A_prep.bat` PDF extractor, ensure you remove:
* Page numbers, running headers, and footers left over from a PDF conversion. These land in the middle of paragraphs and end up inside training data.
* Footnote markers and footnote bodies.
* Tables of contents, copyright pages, "about the author".
* Hard line wrapping at a fixed column. If every line breaks at 72 characters, join the lines within each paragraph first — otherwise every line looks like its own paragraph.
* Hyphenation across line breaks (`consid-\neration`).

Run inspection before processing:
```bash
python 0B_inspect.py
```

It reports block counts, the length distribution, how much direct speech there is, and how many chapters and scene breaks it found — and then tells you what that means for your thresholds. Read its output before running anything else.

Two numbers matter most:
- **Median block length under 200 characters** is normal — those are ordinary paragraphs, and the model does the segmentation.
- **Very little direct speech** is a warning. This pipeline builds dialogue transcripts. A source that is almost entirely narration will yield few rows; lower `convert.min_dialogue_share` and expect step 03 to skip a lot.

**How much text do you need?** There is no hard floor, but under roughly 200 converted transcripts the dataset is thin, and you should lean harder on step 05 (raise `synth.count`) and on the diluent. `tools/status.py` says so explicitly when it sees a small dataset.

---

## Configuration Reference (`config.yaml`)

One config file per run. `config.yaml` in the project root is the template; every value in it has a default in `pipeline/config.py`, so you only keep the lines you actually change.

The settings you will touch first:

| Setting | What it does |
|---|---|
| `lang` | Which language pack to use. Selects `prompts/<lang>.yaml`. |
| `input.path` | Your source text (auto-resolves `.md` or `.txt`). |
| `workdir` | Where intermediate files go. **One per language, never shared.** |
| `llm.base_url` | Your model server. |
| `llm.workers` | `1` for a local single-slot server, `6`–`8` for a hosted API. |
| `llm.max_tokens` | Must comfortably exceed your longest scene, or transcripts get truncated. |
| `convert.char` | `auto`, or a character name to build the dataset around one character. |
| `synth.count` | How many synthetic sessions to generate. |
| `assemble.diluent.enabled` | Off by default. See [The Diluent](#the-diluent-off-distribution-hardening). |

```yaml
# Language pack selection (selects prompts/<lang>.yaml)
lang: en

input:
  # Primary source path (pipeline auto-resolves between .md and .txt)
  path: input/source.txt
  encodings: [utf-8-sig, utf-8, cp1252, latin-1]

# Intermediate files directory
workdir: work/en

llm:
  base_url: http://localhost:5001
  model: local
  api_key: none
  temperature: 0.0
  max_tokens: 4096
  retries: 3
  timeout: 6000
  workers: 1

  # Optional per-step model routing overrides:
  segment:
    base_url: http://localhost:5002
    model: llama-3.2-3b-instruct
    max_tokens: 1024

  convert:
    base_url: http://localhost:5001
    model: qwen2.5-14b-instruct
    max_tokens: 4096

segment:
  # Structural blocks longer than this trigger LLM boundary discovery
  max_block_chars: 14000
  min_scene_chars: 700
  max_scene_chars: 11000
  llm_window_chars: 9000
  llm_window_overlap: 6

attribute:
  temperature: 0.0

convert:
  char: auto
  min_turns: 2
  min_dialogue_share: 0.01
  # Non-fiction dramatization toggle
  allow_informational: true
  temperature: 0.35

validate:
  lexical_overlap_min: 0.55
  llm_check: true
  temperature: 0.0

synth:
  enabled: true
  count: 200
  anchors_per_gen: 2
  anchor_max_chars: 5000
  max_anchor_overlap: 0.5
  dedupe_jaccard: 0.35
  temperature: 0.7
  chars: []

format:
  system_card: ""
  system_card_core: ""
  output: dataset_base.jsonl

augment:
  enabled: true
  variants_per_scene: 3
  keep_original: true
  min_overlap: 0.55
  min_distinctive: 0.70
  temperature: 0.35
  max_axes_per_variant: 3
  axes_file: axes.yaml
  output: dataset_aug.jsonl
  seed: 42
  limit: 0

assemble:
  source_input: auto
  diluent:
    enabled: false
    files: []
    count: 800
    translate: false
    safety_filter: false
  max_session_chars: 16000
  output: dataset_final.jsonl

publish:
  repo_id: ""
  private: true
  eval_share: 0.05
  card_output: DATASET_CARD.md
  license: other
  seed: 42

antislop:
  enabled: false
  base_url: http://localhost:8000
  model: unsloth/Llama-3.2-3B-Instruct
  temperature: 1.0
  max_tokens: 4096
  adjustment_strength: 100.0
  output: dataset_antislop.jsonl
```

---

## Language Packs & Prompt Mechanics (`prompts/en.yaml`)

Prompts are stored externally in `prompts/<lang>.yaml`. Placeholders use double-bracket notation (`[[scene]]`, `[[char]]`, `[[user]]`), avoiding escaping issues with JSON syntax.

### Key Configuration Elements:
* `meta.generic_speakers`: Pronouns and generic descriptors (`someone`, `voice`, `narrator`) filtered out during attribution.
* `meta.chapter_words`: Recognized heading prefixes (`chapter`, `part`, `book`, `section`).
* `cards.full` / `cards.core`: Base system cards injected into final sessions. `cards.core` omits perspective and style rules to prevent collisions with Step 07 formatting blocks.
* `synth.situations`: The situation matrix for Step 05. Edit this list to align with your source domain.

---

## Checkpoints, Quality Audits & Inspection

Audit your data at key stages using the built-in inspection suite:

```bash
# Print stage-by-stage data yield funnel
python tools/status.py

# Review random completed sessions
python tools/inspect_dataset.py --sample 10 work/en/dataset_final.jsonl

# Inspect token distributions and message lengths
python tools/inspect_dataset.py --stats work/en/dataset_final.jsonl

# Scan for repetitive loops or degenerate endings
python tools/inspect_dataset.py --health work/en/dataset_final.jsonl
```

### What Quality Means Here
The pipeline's filters catch mechanical failures: the converter paraphrasing instead of carrying prose across, a rewrite that softens the source, a generator that repeats itself, a model refusing. They cannot tell you whether the dataset is any good.

The statistics look perfectly healthy on a dataset full of subtly flattened rewrites. Reading ten sessions does not:

```bash
python tools/inspect_dataset.py --sample 10 work/en/dataset_final.jsonl
```

Do that before you train, every time. And keep the checkpoints below in mind — especially reading a handful of transcripts side by side with their source scenes after step 03. That single check catches more real problems than every threshold in the config put together.

### Two Habits That Will Save You a Night
1. **Run the offline smoke test after any config change:** `python tools/mock_llm_server.py --port 8999` and `python tools/run_all.py --config tests/config.test.yaml`. It tests the plumbing, not the quality. It will catch a broken config, a malformed language pack, or a missing dependency before you commit real compute.
2. **Run `0B_inspect.py` on your actual file and read its last section:** It reports block counts, length distribution, how much direct speech there is, and how many chapters and scene breaks it found — and then says what that means for your thresholds. Most failures announce themselves there, for free, in one second.

### Critical Human Verification Checkpoints:
1. **After Step 01:** Inspect `work/en/scenes.jsonl`. Verify that scenes represent coherent narrative units rather than fragmented sentences.
2. **After Step 02:** Review scenes with non-zero `unknown_speakers` in `work/en/scenes_attr.jsonl`. Verify that historical figures or key characters are not mislabeled as `unknown`.
3. **After Step 03:** Read 5–10 transcripts side-by-side with source scenes. Confirm that non-fiction scenes are converted into engaging discussions without Turn-1 summary dumps.
4. **After Step 04:** Audit `work/en/rp_rejected.jsonl`. If scenes are rejected for low lexical overlap, adjust `validate.lexical_overlap_min` or refine conversion prompts.

**Alternate Description:**
The pipeline reports a lot. These four checks are the ones that matter, and skipping them is how you find out at the end of a training run that step 02 was broken.

| After | What to check |
|---|---|
| **01** | Scene length distribution. A median far below `segment.min_scene_chars` means the chunker is shredding the text — raise the thresholds. |
| **02** | Open 10–15 scenes with a non-zero `unknown_speakers` and read the attribution. Everything downstream trusts it. |
| **03** | Read 5–10 transcripts side by side with their source scenes. If the narration was **retold** rather than carried over, fix the convert prompt in the language pack. Confirm that non-fiction scenes are converted into engaging discussions without Turn-1 summary dumps. |
| **04** | Read `rp_rejected.jsonl`. A reason that repeats is a step 03 prompt problem, not a reason to throw data away. |

---

## Overnight Execution & Automated Gates

Run the entire pipeline overnight with automated error recovery:

```bash
python tools/run_all.py --from 0A --to 08
```

Between steps, execution passes through automated validation gates:
* **Attribution Gate (Post-Step 02):** Compares text dialogue markers against attributed lines. If attribution losses exceed thresholds, execution halts to avoid corrupting downstream transcripts. *(Note: This gate is automatically bypassed if `convert.allow_informational: true` is set, as non-fiction books frequently use quotes for citations rather than spoken dialogue).*
* **Truncation Gate (Post-Step 03):** Compares transcript length to source scene size. If transcripts are less than 50% of source volume, it flags potential `max_tokens` context exhaustion.

Use `--force` to bypass validation gates if processing highly abstract or non-traditional texts.

---

## Importing Pre-Existing Dialogue

If your source data is already conversational (e.g., chat logs, transcript exports, or ShareGPT archives), bypass Steps 01–05:

```bash
python tools/import_transcripts.py logs/*.txt --char Narrator --user Inquirer
python 06_format.py
python 07_augment.py
python 08_assemble.py
python 09_publish.py --dry-run
```

Supported formats:
* **ShareGPT Files (`.jsonl`, `.parquet`):** Standardizes role mappings and extracts character names from system cards.
* **Plain Text Logs (`Name: Line`):** Segments sessions using `* * *` dividers or multiple blank lines, automatically mapping secondary speakers into narrative context.

---

## Multilingual Execution & Merging

Execute each language in an isolated workdir, then merge the final outputs:

```bash
# Process English
python tools/run_all.py --config config.en.yaml

# Process Second Language
python tools/run_all.py --config config.ru.yaml

# Merge datasets with balanced sampling
python tools/merge_datasets.py dataset_multilingual.jsonl \
    work/en/dataset_final.jsonl work/ru/dataset_final.jsonl --balance
```

The `--balance` flag samples evenly from each language pool, preventing larger corpora from skewing model outputs.

---

## Adding New Languages

1. Copy the template pack:
   ```bash
   cp prompts/en.yaml prompts/de.yaml
   ```
2. Translate all strings, paying close attention to:
   * `meta.chapter_words`: Regional terms for chapters and structural divisions.
   * `meta.generic_speakers`: Regional pronouns and generic role identifiers.
   * `meta.stem_length`: Word prefix length for stemming checks (e.g., 6 for English, 5 for inflected languages like Russian or German).
   * `axes`: Remove formatting axes that do not apply to the target typography.
3. Update `lang: de` in your `config.yaml`.

---

## The Diluent: Off-Distribution Hardening

Training exclusively on a single book can over-index the model on niche vocabulary and specific formatting traits. To retain general conversational capabilities, mix in 20% to 30% general roleplay data:

```yaml
assemble:
  diluent:
    enabled: true
    files: [datasets/general_rp.jsonl]
    count: 500
    translate: false
    safety_filter: true
```

The assembly step normalizes role names, strips external macro markers, enforces character budget constraints, and applies co-occurrence safety filtering.

---

## Behavioral Directives via Presets

Extract diverse behavioral guidelines from SillyTavern preset files to enrich Step 07 formatting variety:

```bash
python tools/extract_axes.py presets/*.json --out axes.yaml
python tools/extract_axes.py --clean
```

Harvested directives (e.g., proactive behavior, avoidance of purple prose, anti-sycophancy) are injected into Step 07 system cards, training the model to follow unseen behavioral instructions at inference time.

---

## Publishing & Local Distribution

Validate, split, and generate documentation locally:

```bash
# Dry run verification
python 09_publish.py --dry-run

# Local production export
python 09_publish.py
```

Outputs are written to your dataset directory:
* `dataset_train.jsonl`: Training split.
* `dataset_eval.jsonl`: Held-out evaluation split (governed by `publish.eval_share`).
* `DATASET_CARD.md`: Comprehensive model card detailing token distributions, message counts, and source provenance.

---

## Fine-Tuning Guidance

* **Loss Calculation (`train_on_inputs: false`):** Configure your trainer (Axolotl, Unsloth, LLaMA-Factory) to mask user turns and system cards out of the loss computation. Loss should only be evaluated on `gpt` turns.
* **Epoch Allocation:** Do not scale epochs linearly with Step 07 dataset expansion. Augmented sessions represent identical underlying knowledge in varied formatting wrappers. Train for **1 to 2 epochs** to prevent overfitting.
* **Evaluation Discipline:** Evaluate exclusively against `dataset_eval.jsonl`. Verify that the model generates in-universe responses to scenarios that never occurred in the source text without falling back on rote memorization.

---

## Troubleshooting & Failure Modes

| Symptom | Underlying Cause | Remediation |
| :--- | :--- | :--- |
| **Step 0A fails on Windows with DLL errors** | Python OCR packages trying to load uncompiled C++ extensions. | Use `0A_prep.bat` (native Docling CLI + `0A_md_splitter.py`). |
| **Step 0A missing input files** | Input document not found in `input/`. | Verify file location. The pipeline auto-detects `input/source.md` and `input/source.txt`. |
| **Step 01 produces huge mega-scenes (10k+ chars)** | `segment.max_scene_chars` set too high. | Lower `segment.max_scene_chars` to `4000` and `segment.max_block_chars` to `4500` to prevent LLM context exhaustion. |
| **Step 01 hangs on a massive block (500k+ chars)** | Heading styles in non-fiction did not match `chapter_words`. | Ensure Markdown `#` headers are present or lower `segment.max_block_chars`. Checkpoints save every window. |
| **Step 02 halts at attribution gate** | Non-fiction source has few spoken lines. | Ensure `convert.allow_informational: true` is set in `config.yaml` to bypass fiction-only attribution gates. |
| **Step 03 invalid JSON on many scenes** | Scenes contain pure OCR checkbox/table noise or token cutoffs. | The pre-flight cleaner (`repair_and_filter_scenes`) automatically routes non-narrative forms to `debug/skipped_garbage_scenes.jsonl` and salvages unescaped quote turns. Inspect `debug/convert_fail_<id>.txt`. |
| **Step 03 Turn 1 is a massive summary dump** | Conversion prompt framed scene as an interview/quiz. | Use the v4 diegetic conversion prompt with anti-dump constraints. |
| **Transcripts are much shorter than their scenes** | `llm.max_tokens` is too low. | Raise `llm.max_tokens` to `4096`. The gate after step 03 in `run_all.py` catches this. |
| **Everything is rejected at step 04 with low overlap** | The converter paraphrased instead of retaining prose. | Read a few rejected transcripts. If they read like a summary, sharpen `convert.system` in the language pack. If they look fine, lower `validate.lexical_overlap_min`. |
| **Step 05 halts with "No usable characters found"** | Documentary source has 0 attributed dialogue lines. | Handled automatically by the fallback cascade (`top_characters` -> scene entities -> `pack.fallback_char`). |
| **Step 05 rejects rows with "too few turns: 1 < 2"** | Model generated only character response or only user prompt. | Automatically repaired via `synthesize_missing_turn`, which generates the missing question or response. |
| **Step 05 fails with "name 'situation' is not defined"** | Scope unpacking mismatch in worker return tuple. | Fixed in v4: `work(row_id)` returns `situation` so completion prompts receive full context. |
| **Step 05 invalid JSON on 100% of rows** | Model responding with conversational chatter without JSON mode. | Ensure `llm.chat(..., json_mode=True)` is used in Step 05. Inspect `debug/synth_fail_<id>.txt`. |
| **Step 07 appears hung during long runs** | Step 07 processes hundreds of jobs sequentially. | Real-time ETA and per-job progress output are enabled; all accepted variants are continuously saved to `aug_accepted_partial.jsonl`. |
| **Step 07 rejects variants as "softened"** | Model replaced concrete rare words with generic descriptions. | Switch to an uncensored base model or adjust `augment.min_distinctive`. If it is a false positive on a heavily inflected language, check `meta.stem_length` in the pack. |
| **Model refusals** | Model triggered safety filters. | Recorded in `refusals_NN.jsonl` and **not** marked done. Point `llm.base_url` at a different model and rerun the step. |
| **The language pack fails to load** | YAML parsing error. | The error names the key and the line. Almost always an unquoted list item containing `: `. |

---

## Scope and Limits

It is worth being blunt about this, because "universal" is not the same as "eats anything".

### Works out of the box (v1)
* **English, Russian:** shipped language packs.
* **Any encoding:** detected and reported; UTF-8 preferred.
* **Chapter headings, scene breaks:** several conventions, plus bare numbers and Roman numerals.
* **Guillemets, straight and curly double quotes, dash-led dialogue, German low quotes, CJK corner brackets:** recognised and interconvertible.
* **Sources with no structural markup at all:** segmented by the model instead; slower, same result.
* **A file with no blank lines at all:** split on sentence borders, then on whitespace, as a fallback.

### Works out of the box (v4)
* **Fiction & Expository Non-Fiction:** Novels, short stories, historical monographs, investigative reports, trial transcripts, and technical case studies.
* **Scanned PDFs & Form Documents:** Native Docling ingestion via Step 0A with automatic base64 image extraction and Markdown heading integration (`#` through `####`).
* **English, Russian:** Shipped, production-validated prompt packs.
* **Any text encoding:** Auto-detected (UTF-8, UTF-8-SIG, CP1251, CP1252, Latin-1).
* **Typography conventions:** Guillemets, straight and curly quotes, dash-led dialogue (`— `), German low quotes, and CJK corner brackets (`「...」`).
* **Unstructured texts:** Sources with no headings or scene breaks are automatically segmented via atomic-checkpointed LLM sliding windows.
* **Texts lacking blank lines:** Automatically hard-split on sentence boundaries, then whitespace fallbacks.

### Works, but you have to write a language pack first
Any language written with spaces: German, French, Spanish, Polish, Turkish, Greek and so on. Copy `prompts/en.yaml`, translate the strings, set the chapter words, the generic speakers and the stem length. No Python is involved.

### Works with configuration adjustments
* **Expository Non-Fiction & Text Without Dialogue:** Originally considered unusable in v1, v4 includes `convert.allow_informational: true` to convert technical manuals, essays, historical monographs, and evidentiary case files into diegetic conversations between an in-universe expert (`meta.fallback_char`) and an active conversational partner (`meta.fallback_user`).
* **Single-quote dialogue & mixed JSON quotes:** Handled natively by v4's multi-tier regex turn salvaging in `pipeline/llm.py` and `03_convert.py`. Single-quoted turns (`'role': 'char'`) and unescaped dialogue quotes within speech are repaired rather than causing JSON decode failures.

### Does not work well, and here is exactly why
* **Chinese, Japanese, Thai, Korean:** While `01_segment.py` natively recognizes CJK sentence terminators (`。！？`), downstream quality gates in `textstats.py` tokenize on space-delimited letter runs (`[^\W\d_]{4,}`). On unspaced scripts, whole clauses are grouped into single tokens, skewing `overlap` and `retention` metrics and rendering 4-gram deduplication ineffective. Full CJK support requires character n-gram tokenization.
* **Right-to-left scripts:** Arabic and Hebrew tokenize into word units, but typography transforms (dash dialogue, punctuation mirroring, and directional marker preservation) have not been tested.

### Wrong tool entirely
* **Screenplays and stage plays:** Formats where speaker names sit on isolated lines above their lines (`ANNA` / `You said midnight.`) without surrounding narrative prose. The pipeline expects continuous narrative prose and direct speech markers. For plays or screenplays, normalize them into `Name: line` format and import them directly via `tools/import_transcripts.py`.
* **Archives that are already dialogue:** Chat logs, exported SillyTavern sessions, or existing ShareGPT files. Running them through segmentation and attribution is redundant and lossy. Use `tools/import_transcripts.py` to bypass Steps 01–05 and enter directly at Step 06.
* **Extremely short sources:** A text producing fewer than ~50 scenes will yield a narrow training distribution. If processing a short text or novella, enable `synth.enabled: true` with a higher target count (`synth.count: 200+`) and mix in general roleplay data via the diluent in Step 08.

---

## Repository Layout

```text
menu.py                     # Main interactive terminal launcher
config.yaml                 # Active configuration profile
0A_prep.bat                 # Primary document ingestion script (Docling CLI)
0A_md_splitter.py           # Decodes base64 images and cleans Markdown links
0B_inspect.py               # (Previously 00) Source text structure & readiness inspection
01_segment.py               # Scene boundary detection with atomic disk checkpointing
02_attribute.py             # Speaker line attribution engine
03_convert.py               # Diegetic roleplay & expository conversion engine
04_validate.py              # Transcript validation & lexical overlap gate
05_synthesize.py            # Grounded synthetic scenario generator
06_format.py                # ShareGPT dataset packaging
07_augment.py               # System prompt format-variation engine
08_assemble.py              # Dataset compilation & diluent integration
09_publish.py               # Local dataset card builder & train/eval splitter
10_antislop.py              # Optional AntiSlop API regeneration pass

pipeline/
  config.py                 # Configuration loader & dual-format (.md/.txt) resolver
  files.py                  # Encoding detection & atomic JSONL I/O
  llm.py                    # OpenAI client with per-step routing & JSON salvage
  lang.py                   # Language pack loader & placeholder rendering
  textstats.py              # Lexical statistics, overlap, retention & n-grams
  axes.py                   # Formatting axes & macro cleansing
  sharegpt.py               # ShareGPT structure validation & turn normalizers
  safety.py                 # Content filters for external datasets

prompts/
  en.yaml                   # English prompt pack & system cards
  ru.yaml                   # Russian prompt pack & system cards

tools/
  run_all.py                # Multi-step automated unattended runner
  status.py                 # Pipeline yield funnel & inspection reports
  inspect_dataset.py        # Dataset health scanner & session viewer
  import_transcripts.py     # Pre-formatted dialogue importer
  extract_axes.py           # Behavioral directive harvester for SillyTavern presets
  merge_datasets.py         # Balanced multilingual dataset merger
  mock_llm_server.py        # Offline local testing mock server

input/                      # Source documents
  images/                   # Extracted base64 images from Step 0A
  source.pdf                # Raw PDF input
  source.md                 # Cleaned Markdown output from Step 0A

work/                       # Intermediate runtime artifacts and logs
  en/
    debug/                  # Step 01 atomic checkpoints & boundary logs
    scenes.jsonl            # Segmented scenes
    scenes_attr.jsonl       # Attributed speaker lines
    rp_raw.jsonl            # Unfiltered converted transcripts
    rp_clean.jsonl          # Validated transcripts
    rp_rejected.jsonl       # Transcripts failing overlap gates
    synth_clean.jsonl       # Validated synthetic sessions
    dataset_base.jsonl      # Step 06 output
    dataset_aug.jsonl       # Step 07 output
    dataset_final.jsonl     # Assembled training dataset
    dataset_train.jsonl     # Training split
    dataset_eval.jsonl      # Evaluation split
    DATASET_CARD.md         # Final dataset documentation card
    dataset_antislop.jsonl  # (Optional) Step 10 de-slopped dataset
```

---

## Anchors Note

In `config.yaml`, the recommended setting for **`anchors_per_gen`** depends on the size of your local model and the length of your segmented scenes:

### Recommended Setting: `1` (for 12B/14B models)

```yaml
synth:
  enabled: true
  count: 200
  anchors_per_gen: 1         # <--- Set to 1
  anchor_max_chars: 2500     # Keep at 2500–3000 chars
```

### Why `1` is Best for 8B–14B Models (like Mag-Mell 12B)

- **Prevents Prompt Bloat & Generation Degradation**:
   - If `anchors_per_gen: 2` and `anchor_max_chars: 5000`, the prompt alone can inject **10,000+ characters (~2,500+ tokens)** of raw reference text into every synthetic prompt.
   - Mid-sized models struggle to generate structured JSON turns when forced to attend over massive context prompts, often cutting off mid-turn or dropping into conversational preambles.
   - Setting it to `1` with `anchor_max_chars: 2500` feeds a clean, focused ~600-token stylistic anchor, giving the model plenty of room to output complete turns.

- **Reduces False Rejections from `max_anchor_overlap`**:
   - Step 05 checks lexical overlap between the generated dialogue and the anchors:
     ```python
     if anchor_overlap > synth["max_anchor_overlap"]:
         reject("copies the anchors")
     ```
   - When you show the model 2 or 3 anchor scenes (covering thousands of unique vocabulary words), the chance of the synthetic dialogue sharing common content words with *any* of those scenes increases drastically, triggering **`reject: copies the anchors`**.
   - With `1` anchor per generation, the model only needs to avoid quoting that single specific scene.

- **When to use `2` (the upstream default)**:
   - Only use `anchors_per_gen: 2` if you are running a **70B+ model** or a hosted frontier API (e.g., Claude 3.5 Sonnet / Qwen 2.5 72B) with large context windows where the character needs to synthesize reasoning patterns across multiple distinct scenes.