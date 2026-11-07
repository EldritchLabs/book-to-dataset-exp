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
import sys
import time
import requests

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl

def regenerate_session(session, cfg):
    """Replays a session through the AntiSlop API to regenerate GPT turns."""
    anti_cfg = cfg.get("antislop", {})
    api_url = anti_cfg.get("base_url", "http://localhost:8000").rstrip("/")
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
                "temperature": anti_cfg.get("temperature", 1.0),
                "max_tokens": anti_cfg.get("max_tokens", 4096),
                "adjustment_strength": anti_cfg.get("adjustment_strength", 100.0),
                "antislop_enabled": True,
                "stream": False
            }
            
            # Call the AntiSlop API
            for attempt in range(3):
                try:
                    resp = requests.post(f"{api_url}/v1/chat/completions", json=payload, timeout=6000)
                    resp.raise_for_status()
                    new_text = resp.json()["choices"][0]["message"]["content"]
                    break
                except Exception as e:
                    print(f"  [warn] AntiSlop API error: {e}. Retrying in 5s...", file=sys.stderr)
                    time.sleep(5)
            else:
                raise RuntimeError("AntiSlop API failed after 3 attempts.")
            
            messages.append({"role": "assistant", "content": new_text})
            new_conversations.append({"from": "gpt", "value": new_text})
            
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
            
            clean_session = regenerate_session(session, cfg)
            append_jsonl(out_path, clean_session)
            print("OK", flush=True)
    except KeyboardInterrupt:
        print("\n[Interrupted] Progress saved. You can resume later.")
        sys.exit(130)
        
    print(f"\nDone! AntiSlop dataset saved to: {out_path}")

if __name__ == "__main__":
    main()