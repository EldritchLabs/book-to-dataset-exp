# -*- coding: utf-8 -*-
"""Step 10 — AntiSlop Final Polish (Optional).

This step connects to Sam Paech's AntiSlop API server to regenerate the
assistant responses in the final dataset, actively backtracking and
downregulating slop phrases (e.g., "testament to", "shivers down my spine").

Prerequisite:
You must have the antislop-sampler running locally in a separate terminal.
  git clone https://github.com/sam-paech/antislop-sampler.git
  cd antislop-sampler
  pip install fastapi uvicorn ipywidgets IPython transformers bitsandbytes accelerate
  python run_api.py --model <your_model> --slop_adjustments_file slop_phrase_prob_adjustments.json

    Input:  <workdir>/dataset_final.jsonl
    Output: <workdir>/dataset_antislop.jsonl

    python 10_antislop.py [--config config.yaml]
"""
import os
import re
import sys
import time
import requests

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl

def regenerate_session(session, cfg, dpo_out_path=None):
    """Replays a session through the AntiSlop API to regenerate GPT turns."""
    anti_cfg = cfg.get("antislop", {})
    api_url = anti_cfg.get("base_url", "http://localhost:5001").rstrip("/")
    model = anti_cfg.get("model", "unsloth/Llama-3.2-3B-Instruct")
    
    messages = []
    new_conversations = []
    
    for msg in session.get("conversations", []):
        if msg["from"] == "system":
            messages.append({"role": "system", "content": msg["value"]})
            new_conversations.append(msg)
        elif msg["from"] == "human":
            messages.append({"role": "user", "content": msg["value"]})
            new_conversations.append(msg)
        elif msg["from"] == "gpt":
            payload = {
                "model": model,
                "messages": messages,
                "temperature": anti_cfg.get("temperature", 0.5),
                "max_tokens": anti_cfg.get("max_tokens", 4096),
                "adjustment_strength": anti_cfg.get("adjustment_strength", 25.0),
                "antislop_enabled": True,
                "stream": False
            }
            
            # Call the AntiSlop API
            new_text = None
            orig_text = msg["value"]
            
            # Extract named entities from system card as session whitelist
            sys_val = next((m["value"] for m in session.get("conversations", []) if m["from"] == "system"), "")
            session_whitelist = list(set(re.findall(r'\b[A-Z][a-zA-Z’\'-]{2,}\b', sys_val)))
            if session_whitelist:
                payload["whitelist"] = session_whitelist[:50]

            for attempt in range(3):
                try:
                    resp = requests.post(f"{api_url}/v1/chat/completions", json=payload, timeout=6000)
                    resp.raise_for_status()
                    cand = resp.json()["choices"][0]["message"]["content"].strip()
                    
                    # N-Gram Degeneration Check: detect repeating phrase loops
                    tokens = cand.lower().split()
                    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
                    repetition_found = False
                    if trigrams:
                        from collections import Counter
                        top_tg = Counter(trigrams).most_common(1)[0]
                        if top_tg[1] >= 4 and len(tokens) > 30:
                            repetition_found = True

                    if repetition_found:
                        print(f"  [warn] AntiSlop repetition loop detected on attempt {attempt+1}, adjusting...", file=sys.stderr)
                        payload["temperature"] = max(0.5, float(payload["temperature"]) - 0.2)
                        payload["adjustment_strength"] = max(10.0, float(payload["adjustment_strength"]) * 0.5)
                        continue

                    new_text = cand
                    break
                except Exception as e:
                    print(f"  [warn] AntiSlop API error: {e}. Retrying in 3s...", file=sys.stderr)
                    time.sleep(3)

            # Fall back safely to original text if API fails or enters repetition loops
            final_turn = new_text if (new_text and len(new_text) >= 20) else orig_text
            if dpo_out_path and final_turn != orig_text:
                prompt_ctx = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
                append_jsonl(dpo_out_path, {
                    "prompt": prompt_ctx,
                    "chosen": final_turn,
                    "rejected": orig_text,
                })
            messages.append({"role": "assistant", "content": final_turn})
            new_conversations.append({"from": "gpt", "value": final_turn})
            
    return {"conversations": new_conversations}

def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    anti_cfg = cfg.get("antislop", {})
    
    if not anti_cfg.get("enabled", False):
        print("antislop.enabled is false — skipping Step 10.")
        return

    source_path = cfg["assemble"]["output"]
    if not os.path.exists(source_path):
        sys.exit(f"\n{source_path} not found. Run 08_assemble.py first.\n")

    out_path = work_path(cfg, anti_cfg.get("output", "dataset_antislop.jsonl"))
    
    sessions = read_jsonl(source_path)
    existing = read_jsonl(out_path)
    done = len(existing)
    
    if done > 0:
        print(f"[resume] {done} sessions already de-slopped, continuing...")
        
    pending = sessions[done:]
    print(f"To process: {len(pending)} sessions via AntiSlop API ({anti_cfg.get('base_url')})")
    
    if not pending:
        print("Nothing to do.")
        return

    # The AntiSlop API does not support concurrency, so we process sequentially.
    start_time = time.time()
    try:
        for idx, session in enumerate(pending, start=1):
            curr_num = done + idx
            pct = (curr_num / len(sessions)) * 100
            
            elapsed = time.time() - start_time
            rate = elapsed / idx
            rem_sec = int(rate * (len(sessions) - curr_num))
            eta_str = f"{rem_sec // 60}m {rem_sec % 60}s" if rem_sec >= 60 else f"{rem_sec}s"

            gpt_turns = sum(1 for m in session.get("conversations", []) if m.get("from") == "gpt")
            print(f"[{curr_num}/{len(sessions)} {pct:5.1f}% | ETA {eta_str}] Regenerating session {curr_num} ({gpt_turns} turns)...", end=" ", flush=True)
            
            dpo_path = work_path(cfg, "dataset_antislop_dpo.jsonl") if anti_cfg.get("save_dpo_pairs", False) else None
            clean_session = regenerate_session(session, cfg, dpo_out_path=dpo_path)
            append_jsonl(out_path, clean_session)
            print("OK", flush=True)

            debug_dir = work_path(cfg, "debug")
            trace_log = os.path.join(debug_dir, "stage_10_trace.log")
            with open(trace_log, "a", encoding="utf-8") as tf:
                tf.write(f"[STAGE 10][PROCESSED] session={curr_num} gpt_turns={gpt_turns}\n")
    except KeyboardInterrupt:
        print("\n[Interrupted] Progress saved. You can resume later.")
        sys.exit(130)
        
    total_elapsed = max(0.1, time.time() - start_time)
    print(f"\nDone! AntiSlop dataset saved to: {out_path} "
          f"({len(pending)} sessions in {total_elapsed:.1f}s, "
          f"{len(pending) / (total_elapsed / 60):.1f} sessions/min)")

if __name__ == "__main__":
    main()