## v3 to v4 Change Log

#### 1. Updated `pipeline/llm.py`
Upgrade `extract_json()` with heuristic repair logic:
- Truncate preambles up to the first `{` or `[`.
- Strip markdown fences and trailing text after the last `}` or `]`.
- Auto-close truncated JSON strings, objects, and lists.
- Fix common syntax flaws (trailing commas before closing brackets, unescaped newlines inside strings).
- Provide a safe empty fallback `{}` if the input contains zero direct speech/characters rather than throwing an exception.

#### 2. Updated `02_attribute.py`
Incorporate a polymorphic unpacker and an automatic retry/fallback for non-narrative documentary forms (e.g., clearance reports, inventory logs, OCR tables) where the model fails to find spoken dialogue.

#### 3. Updated `prompts/en.yaml` (Step 02 Retry Robustness)
Update the `attribute.user` prompt in `prompts/en.yaml` to explicitly tell the model how to handle expository documents and form scans with no dialogue:

### Patch Set & Diagnosis: Why Step 03 Emits "invalid JSON" in Bulk

4. **Unescaped Quotes Inside Generated Dialogue:** When a 12B model produces dialogue containing straight quotes or contractions (`"text": "He muttered, "No way," before turning."`), standard `json.loads` chokes immediately.
5. **Context Window Token Cuts:** Multi-turn conversational dramatizations can easily span 2,000–4,000 tokens. When `max_tokens` cuts off generation mid-turn, the JSON closes abruptly without closing quotes `"` or brackets `]}`.
6. **No Forensic Traceability:** When `extract_json()` fails, the raw reply vanishes into `stderr` without being logged to disk. You cannot see whether it was a token truncation, a markdown fence mishap, or a syntax error.
7. **Missing Heuristic Turns Extraction:** Even if the JSON syntax is completely broken, the text itself usually contains clear turn structures like `{"role": "user", "text": "..."}` or `[user] ... [char] ...` that can be salvaged via regex without losing GPU time.

#### 4. Upgrade `pipeline/llm.py`
Add robust turn-level salvage and write failure dumps into a dedicated debug folder when all parsing fails.

#### 5. Upgrade `03_convert.py`
Add:
- Automatic saving of every unparseable raw model response to `work/<lang>/debug/convert_fail_<scene_id>.txt` so failures can be inspected immediately.
- A secondary regex fallback inside `03_convert.py` that reconstructs dialogue turns from semi-structured markdown/free-text if JSON parsing fails.
- Fallback role alternation guarantee.

#### 6. Upgrade `02_attribute.py` (Universal Diagnostic Logging)
Ensure Step 02 also saves any failed raw model output to `work/<lang>/debug/attr_fail_<scene_id>.txt` so broken scene attributions can be inspected rather than silently failing.

#### 7. Update `04_validate.py` (Validation Gate Salvage & Logging)
Ensure Step 04 records invalid judge responses to `work/<lang>/debug/judge_fail_<rp_id>.txt` rather than leaving unexplained gaps.

### What Changes After Applying These Chunks
- **Automatic JSON Salvaging:** Any scene where the 12B model produces valid conversational turns inside broken, cut-off, or unescaped JSON will now be salvaged by regex parsing in `pipeline/llm.py` and `03_convert.py` instead of aborting.
- **Inspectable Failure Dumps:** Any scene that genuinely cannot be parsed will write the raw LLM output into `work/en/debug/convert_fail_<scene_id>.txt` alongside the exact error message.
- **Resumption:** Simply re-run Step 03 from the menu (`3`) or via command line (`python 03_convert.py`). The script resumes from where it left off, and salvaged outputs will convert cleanly.

---

### Why Scenes Like 5 Are Failing
Looking directly at your samples for `scene 2`, `scene 3`, and `scene 5`:
1. **Unusable OCR Garbage & Massive Noise Ratio**: `scene 5` is 3,124 characters of pure OCR checkbox soup (`O O ALTERED STOCK`, `CHE - TEHS-`, `OE HU 0 0`). There are no continuous narrative sentences or coherent stories.
2. **Context Window Saturation on Rubbish**: When the 12B model receives 3,000+ characters of fragmented OCR form tokens, it either hallucinates giant, bizarre text blocks that breach `max_tokens` (cutting off before closing JSON), or it outputs invalid quotes, raw unescaped newlines, or conversational chat preambles like `Here is the transcript:\n{"turns": ...`.
3. **Broken / Duplicate ID Lines in `scenes_attr.jsonl`**: The input lines in `scenes_attr.jsonl` contain unescaped characters, missing structural keys, or invalid types that throw exceptions before conversion can even begin.

### Solution
We add `repair_and_filter_scenes(cfg, debug_dir)` to run at the very beginning of `main()` in `03_convert.py`. It:
1. **Validates & Repairs JSON Lines**: Fixes malformed JSON objects in `scenes_attr.jsonl`, ensuring required fields (`id`, `text`, `lines`, `characters`, `dialogue_share`) exist with proper types.
2. **Cleans Forensic / OCR Noise**: Strips out markdown image tags (`![Image](...)`), OCR check-mark runs (`O O`, `0 0`, `[ ]`), and excessive whitespace.
3. **Entropy / Quality Filter**: Discards pure checkbox/form/table noise that has near-zero word density. Sending 3,000 characters of `O O` to the model guarantees invalid JSON, context truncation, or severe hallucination.
4. **Writes an Auditable Repaired File**: Saves valid scenes to `<workdir>/scenes_attr_repaired.jsonl` and dumps filtered-out scenes to `<debug_dir>/skipped_garbage_scenes.jsonl` with explicit reasons.
5. **Upgrades Model Response Turn Extraction**: Handles models wrapping turns in `[char]...[user]...`, Markdown code blocks, or single quotes, ensuring valid generations are never discarded.

### Patch Chunks for `03_convert.py`
#### 8. Add `repair_and_filter_scenes` and helper imports
Add `json` import and the complete repair/cleaning engine before `pick_pair`.

#### 9. Update `main()` in `03_convert.py`
Call `repair_and_filter_scenes()` before loading pending tasks, and expand the turn extraction salvage logic to catch turns formatted with Markdown or alternate brackets.

#### 10. Enhance Response Salvage in `03_convert.py`
Replace the turn-extraction block inside the `parallel_map` worker loop to catch broken quotes, unescaped JSON strings, and turn patterns emitted as lists or raw labeled turns.

### What Happens When You Run Step 03 Now
- `repair_and_filter_scenes()` scans `scenes_attr.jsonl`.
- Severe non-narrative checkbox runs (such as scenes 3, 4, and 5) get logged to `work/en/debug/skipped_garbage_scenes.jsonl` rather than sent to the LLM to fail.
- Useful scenes (like 1, 7, 9, 11, 14, 17) have base64/image tags and OCR noise stripped before prompting, preventing token exhaustion and quote corruption.
- If a response suffers slight JSON syntax breakage, the 3-tier salvage logic reconstructs the dialogue turns, drastically increasing conversion yield.

#### 11. **`05_synthesize.py` Fatal Exit on Non-Fiction**:
- `top_characters()` expects character names with 5+ attributed dialogue lines. In scanned documents, some characters have few or no dialogue lines, and most scenes have `characters: []`. Step 05 will call `sys.exit("\nNo usable characters found...")`.
- **Fix:** If no characters meet the threshold, fall back automatically to `synth.chars`, `pack.fallback_char`, or names extracted from `characters` lists across scenes rather than hard-exiting.
- Also add response salvage and failure logging into `debug/synth_fail_<id>.txt` so 12B JSON formatting issues don't waste synthetic budget.
- Fix character fallback so it doesn't crash when lines are 0.
- Add multi-tier turn salvaging and failure logging for Step 05.

#### 12. **`06_format.py` Empty Dataset Assertion & Card Missing Core**:
- In Step 06, if synthetic rows were disabled or if an empty turn slipped through, formatting can crash or drop rows.
- Ensure scene opener selection handles empty character/user fields gracefully.
- Ensure missing character names fall back to language pack defaults and consecutive human turns are merged.

#### 13. **`07_augment.py` LLM Rewrite Failure on 12B & Token Truncation**:
- Step 07 runs `llm.chat(..., json_mode=True)` and does a bare `extract_json(raw)`. Just like Step 03, a 12B model rewriting full scenes will occasionally hit unescaped quotes or formatting quirks.
- **Fix:** Integrate the multi-tier turn salvage and debug dump logging so rewritten variants are not discarded as simple `"llm error"`.
- Add turn-level salvage and logging to `07_augment.py` so rewritten variants that hit slight JSON formatting hiccups are still salvaged.

#### 14. **`08_assemble.py` Empty Own-Data Assertion**:
- If `augment.enabled` was toggled or if Step 07 output had an alternate name, `resolve_source_input` needs to smoothly locate either `dataset_aug.jsonl` or `dataset_base.jsonl`.
- Make `resolve_source_input` check relative paths and the `workdir` folder so it doesn't fail if the working directory prefix is placed in different config locations.

### Summary
Applying these chunks now will allow the overnight runner (`tools/run_all.py`) to transition directly from Step 03 into Steps 04, 05, 06, 07, and 08 without crashing due to missing character names or unhandled JSON output shapes.

### Root Causes: Why Only 4 Transcripts & 100% Synthesis Failure Occurred
A. **Massive Over-Sized Scenes (11,000–16,000 chars)**:
   In your log:
   ```text
   Done: 10 scenes -> work/en\scenes.jsonl
   Scene length: min 29, median 11431, max 15993
   ```
   `Arthur.md` contains 89,240 characters total, but it was sliced into only **10 giant mega-scenes**, with a median of 11,431 characters! Asking a 12B model to ingest an 11,000-character input scene and output a complete multi-turn JSON roleplay transcript with narration in one go exceeds its generation capacity. It runs out of tokens, cuts off mid-string, and fails parsing on scenes 3, 5, 6, 7, and 8.
   
B. **`is_usable_content` Aggressive Filter in Step 03**:
   `Arthur.md` is pure literary prose, but `is_usable_content` threw out 1 of the 10 scenes (`1 non-narrative forms skipped`) because the heuristic was tuned for document OCR tables rather than literature.

C. **`100% Failure in Step 05 (Synthesis)`**:
   In `05_synthesize.py`, `llm.chat(system, prompt, temperature=...)` was called without `json_mode=True`. The model (Mag-Mell 12B) replied with conversational chatter (`Here is a roleplay session: ...`) or markdown text instead of pure raw JSON. The fallback regex in Step 05 only looked for standard double quotes (`"role": "..."`), which failed if the model used single quotes, unescaped quotes, or plain dialogue lines.

D. **Regex Bug in `llm.py` `extract_json()`**:
   In `extract_json()`:
   ```python
   turn_matches = re.findall(
       r'\{\s*"role"\s*:\s*"(char|user)"\s*,\s*"text"\s*:\s*"(.*?)(?<!\\)"\s*\}',
       text,
       re.DOTALL
   )
   ```
   If the LLM output contained unescaped quotes inside the text (e.g., `"text": "He said, "Yes", and left."`), the regex broke on the first unescaped quote and failed to salvage the turns.

### Patch Set

#### 15. Update `pipeline/config.py` (and your `config.yaml`)
- Cut scenes into manageable, bite-sized narrative chunks (2,000 to 4,500 characters) so that the 12B model never truncates or exhausts its output token limit:
- Also verify your local `config.yaml` has `max_scene_chars: 4000` and `max_block_chars: 4500`.

#### 16. Update `pipeline/llm.py`
- Make `extract_json()` resilient to unescaped internal quotes and conversational prefixes:

#### 17. Update `03_convert.py`
- Remove the aggressive filter so literature like Machen's is **never** dropped, and ensure full prose coverage.
- In `work()` inside `03_convert.py`, ensure `json_mode=True` is always requested so the model adheres to valid JSON.

#### 18. Update `05_synthesize.py`
- Enable `json_mode=True` in `05_synthesize.py` and implement wide regex matching for speaker turns:

### Why Synthetic Generations Return Only 1 Turn

When generating synthetic sessions under high temperature or with specific models, two common partial-generation failure modes occur:
- **Answer Only (`[char]` without `[user]`)**: The model skips generating the user's opening line and jumps straight into the character's monologue, actions, or direct answer.
- **Question Only (`[user]` without `[char]`)**: The model outputs an interlocutor prompt (e.g. asking an uncomfortable question) but stops or gets truncated before emitting the character's response.

Rather than rejecting these generations with `too few turns: 1 < 2`, we can dynamically salvage the session:
- If only `[char]` is produced: Call a fast reverse-turn completion prompt asking the LLM to invent an authentic, reactive `[user]` question or action beat that logically prompted that exact answer.
- If only `[user]` is produced: Call the LLM to generate the `[char]` response grounded in the character's thoughts and anchor scenes.
- If the session ends on an trailing `[user]` turn with multiple turns, drop or complete the trailing prompt so the training session never ends on an unanswered user turn.

#### 19. Add `synthesize_missing_turn()` helper function
Place this helper above `main()` in `05_synthesize.py`. It inspects single-turn outputs and completes the conversational pair.

#### 20. Hook `synthesize_missing_turn()` into the generation loop
In `05_synthesize.py`, if `len(turns) == 1`, invoke the repair function before applying the turn count gate.

### How It Operates

- **When the model only returns a character reply**:
   - `synthesize_missing_turn` reads the character's generated reply and the situation.
   - It issues a targeted prompt to write the user's introductory line or question.
   - The result is prepended as `{"role": "user", "text": "..."}`, creating a valid `user -> char` pair.

- **When the model only returns a user question**:
   - It prompts the character model using the anchor excerpt to generate the in-character answer.
   - The result is appended as `{"role": "char", "text": "..."}`.

- **Validation & Training Integrity**:
   - The resulting session has at least 2 alternating turns (`user` then `char`).
   - It satisfies ShareGPT invariants (starts on user, has an assistant response, doesn't end on user) and passes into Step 06 without data loss.

### Overview of GUI / TUI Enhancements for `menu.py`
To upgrade `menu.py` from a bare text dump into a modern, production-grade terminal dashboard (TUI), we can introduce:

21. **Structured Box-Drawing & Visual Cards**: Replace plain dashed lines (`----------------`) with clean Unicode borders (`┌─┐`, `│`, `└─┘`), distinct header cards, and aligned columns.
22. **Dynamic Progress Bar**: Calculate the total completion percentage across all active pipeline steps and render an animated ASCII/Unicode progress bar (`[████████░░░░] 60% (6/10)`).
23. **Live Endpoint Health Ping**: Test and display the live status of the LLM model endpoint (`http://localhost:5001`) with green `[ONLINE]` or red `[UNREACHABLE]`, including server response latency in milliseconds.
24. **Active Checkpoint Detection**: Inspect `<workdir>/debug/` to see if Step 01 or Step 03 has live sliding-window checkpoints on disk, displaying resume offsets directly on the dashboard.
25. **Color-Coded Status Badges**: High-contrast indicators for completed (`[ DONE ]`), partially completed (`[ PART ]`), pending (`[ TODO ]`), and diagnostic steps (`[ DIAG ]`).
26. **Graceful ASCII Fallback**: Auto-detect non-UTF8 environments or Windows legacy consoles and fall back cleanly without Unicode character corruption.

#### 27. Add Endpoint Ping and Enhanced Palette
Upgrade `Palette` with additional formatting styles and add `check_endpoint_health()` using standard library `urllib`.

#### 28. Enhanced Step Status & Checkpoint Recovery Detection
Inspect step statuses to detect live debug checkpoints (e.g. from Step 01 boundary detection) and compute overall pipeline progress.

#### 29. Overhaul `render()` with Modern Dashboard Layout
Replace the plain rendering function with aligned box-drawing cards, endpoint health indicators, progress bar, and visual badges.

The menu will now render with Unicode box borders, ping your LLM server, calculate your pipeline completion progress, and display your dataset creation progress in a visual terminal dashboard.

### Why The Table Borders Are Misaligned

The misalignment and broken vertical bars happen for two main reasons:
- **ANSI Escape Sequences Count as String Length**:
   When you apply color with `colour(f"[{mark}]", *styles)` or `colour("FOUND", ...)`, the string gets wrapped in invisible terminal escape codes (e.g. `\033[38;5;82m...\033[0m`). Python's string formatting (`{badge:<18}`) measures the **total length including the invisible escape characters**, which shifts columns to the left or right depending on the length of the ANSI sequence.
- **Missing Unicode Glyphs / Wide Characters in Windows `cmd.exe`**:
   The screenshot shows `[? DONE]`, `[? PART]`, and missing lightning bolts `? BOOK-TO-DATASET` because standard Windows `cmd.exe` raster/TrueType fonts (Consolas, Lucida Console) do not support `⚡`, `✔`, `◐`, or `ℹ` in standard codepage 437, turning them into fallback question-mark boxes and throwing off column widths.

---

### Solution

30. Add a helper function `strip_ansi(text)` and `ansi_pad(text_with_ansi, width)` that pads strings based solely on **visible length**.
31. Replace unsupported Unicode glyphs with universal clean ASCII/extended-ASCII symbols (`[DONE]`, `[PART]`, `[TODO]`, `[DIAG]`, `[SKIP]`, and ASCII progress bars `[####....]`) so it renders symmetrically across all Windows and Linux consoles without font corruption.
32. Fix the fixed-width table column arithmetic so the enclosing box border `│` lands on the exact same column for every single row.

#### 33. Add ANSI-aware length calculation and ASCII-safe markers

#### 34. Clean Marks & Overhaul `render()` with Exact Column Math
Replace the markers and table formatting with `pad_visible` to ensure borders align seamlessly:

### What it Looks Like in `cmd.exe` After This Patch

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

---

### Why This Upgrade Is Needed

1. **`07_augment.py` (540 Jobs with 1 Worker)**:
   - In your run, Step 07 has queued **540 LLM rewrite jobs**. With 1 worker on a local model server, this takes considerable time.
   - Currently, it only prints progress every 25 jobs (`processed % 25 == 0`), keeps all accepted sessions purely in an in-memory list (`accepted = []`), and only writes to disk at the very end (`write_jsonl(settings["output"], accepted)`).
   - If interrupted, timed out, or crashed on job 512, **all 512 completed rewrites are lost**, and running Step 07 again starts from job 0.
   - **Fix**: Add atomic, per-job appending to a temporary/partial cache file (`aug_accepted.jsonl`) with live progress percentages, elapsed time/ETA, and instantaneous resume support across restarts.

2. **`10_antislop.py` Sequential Regeneration**:
   - Step 10 processes sequentially across potentially hundreds of multi-turn sessions.
   - It prints `Processing session i/N... OK` but lacks percentage completion, turn count stats, ETA estimation, and error logging when a session fails.
   - **Fix**: Add formatted progress bars, turn tracking, elapsed time/ETA, and atomic checkpoints.

#### 35. Add Timer and Import `time`
Add `time` to the top imports of `07_augment.py`:

#### 36. Implement Real-Time Progress, Incremental Checkpoints & Resume in `rewrite()`
Replace the batch collection loop in `07_augment.py` with an atomic append checkpoint so every accepted session is written immediately to `aug_accepted_partial.jsonl`, enabling immediate resume and live percentage/ETA tracking.

#### 37. Enhance Progress, Turn Metrics & Live ETA in Step 10:
Update `10_antislop.py` so it displays live completion percentage, assistant turns processed, time elapsed, and ETA.

### What You Will See When Running

#### In Step 07 (`07_augment.py`):
```text
Total jobs: 540 | Pending: 540 | Workers: 1

  [1/540  0.2% | ETA 45m 12s] row 0 variant: OK (retention 0.88)
  [2/540  0.4% | ETA 44m 30s] row 0 variant: OK (retention 0.91)
  [3/540  0.6% | ETA 44m 02s] row 0 variant: REJECT (softened 0.62)
  [4/540  0.7% | ETA 43m 48s] row 1 variant: OK (retention 0.84)
```
- Every accepted job is appended immediately to disk (`work/en/aug_accepted_partial.jsonl`).
- If you stop or restart, it prints `[resume] N jobs already completed, M remaining` and instantly skips what has already finished.

#### In Step 10 (`10_antislop.py`):
```text
To process: 200 sessions via AntiSlop API (http://localhost:8000)
[1/200   0.5% | ETA 18m 20s] Regenerating session 1 (3 turns)... OK
[2/200   1.0% | ETA 17m 55s] Regenerating session 2 (4 turns)... OK
```

**Here are the patch chunks for `README.md` to document the newest upgrades:**
1. **Step 07 Incremental Checkpointing & Live ETA**: Documents the transition from memory-only collection to atomic disk appending (`aug_accepted_partial.jsonl`) with live percentage/ETA tracking.
2. **Step 10 Enhanced Metrics & Turn Tracking**: Documents progress calculations, turn counts, and ETA estimation.
3. **Menu Unicode-to-ASCII Alignment**: Documents the `pad_visible` fix and ANSI-safe border rendering.
4. **Step 05 `situation` Unpack Fix**: Details crash resilience in the synthetic generation worker loop.

#### 38. Fix Indentation of `print`, `start_time`, and loop alignment
Fixes the indentation error where `print(Total jobs)` was stranded inside `work()` after `return`.

#### 39. Fix Final Output Compilation & Original Duplication
Ensure that `write_jsonl(settings["output"], ...)` only writes clean ShareGPT objects and does not duplicate original sessions when resuming or rebuilding.

#### 40. Fix Stale Partial Cache Collision
If `dataset_aug.jsonl` does not exist but `aug_accepted_partial.jsonl` does, warn the user and cleanly handle the resume vs fresh start.

#### 41. Anti-Data-Dump Constraint
We will add a new tunable under `convert` in `pipeline/config.py` and `config.yaml`:
```yaml
convert:
  # Options: "on" (strict progressive disclosure), "off" (allows data dumping), "random" (Bernoulli seed mix)
  anti_dump: "random"
```

- **Deterministic Bernoulli Draw**:
   - In `RANDOM` mode, rather than an arbitrary non-reproducible flip, we draw a deterministic pseudo-random flag per scene using `stable_rng(cfg.get("seed", 42), scene["id"])`.
   - In `ON` mode, rule 1 is strictly injected.
   - In `OFF` mode, rule 1 is omitted or relaxed, permitting rich, exhaustive exposition in Turn 1 if the model prefers it.
- **Dynamic Prompt Assembly**:
   - `03_convert.py` inspects `anti_dump` mode for each informational scene and modulates Rule 1 accordingly.

#### 42. Cinematic Beats Weaver
We introduce **`cinematic_beats`** under `convert` (matching `anti_dump`):
- `on`: Strictly mandates physical actions, gestures, environmental interaction, and tone shifts.
- `off`: Dialogue-focused exposition without forced physical stage directions or descriptive action beats.
- `random`: Reproducible Bernoulli coin-flip per scene ID (`stable_rng(..., "cinematic_beats", scene["id"])`) to naturally vary between atmospheric, descriptive roleplay turns and clean direct dialogue exchanges.