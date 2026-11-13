# -*- coding: utf-8 -*-
"""Live vLLM & AntiSlop-vLLM-fork /metrics monitor.

Polls the Prometheus /metrics endpoint on llm.base_url (and antislop.base_url
if enabled) every 2 seconds and displays prompt tok/s, generation tok/s,
KV-cache load %, running/waiting requests, and antislop ban/backtrack rates.

    python tools/metrics_watch.py [--config config.yaml]
"""
import os
import sys
import time
from urllib.parse import urlsplit

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config, parse_args  # noqa: E402


def metrics_url_for(api_url):
    if not api_url:
        return None
    parts = urlsplit(str(api_url).strip())
    if not parts.scheme or not parts.netloc:
        return None
    # return f"{parts.scheme}://{parts.netloc}/metrics"
    return f"{parts.scheme}://{parts.netloc}"


def parse_prom(text):
    samples = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        space = line.rfind(" ")
        if space <= 0:
            continue
        head, val_str = line[:space], line[space + 1:]
        brace = head.find("{")
        name = head[:brace] if brace != -1 else head
        try:
            val = float(val_str)
        except ValueError:
            continue
        samples[name] = samples.get(name, 0.0) + val
    return samples


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)

    urls = []
    llm_m = metrics_url_for(cfg.get("llm", {}).get("base_url"))
    if llm_m:
        urls.append(("LLM", llm_m))
    anti_cfg = cfg.get("antislop", {})
    if anti_cfg.get("enabled"):
        anti_m = metrics_url_for(anti_cfg.get("base_url"))
        if anti_m and anti_m != llm_m:
            urls.append(("AntiSlop", anti_m))

    if not urls:
        sys.exit("No valid base_url configured in config.yaml.")

    prev = {}
    print(f"Watching {', '.join(u for _, u in urls)} (Ctrl+C to stop)\n")
    try:
        while True:
            now = time.time()
            lines = [f"[{time.strftime('%H:%M:%S')}]"]
            for label, url in urls:
                try:
                    resp = requests.get(url, timeout=1.5)
                    resp.raise_for_status()
                    s = parse_prom(resp.text)
                except Exception as exc:
                    lines.append(f"  {label:<10} OFFLINE ({exc})")
                    continue

                p_tok = s.get("vllm:request_prefill_kv_computed_tokens_sum") or s.get("vllm:prompt_tokens_total", 0.0)
                p_sec = s.get("vllm:request_prefill_time_seconds_sum", 0.0)
                g_tok = s.get("vllm:generation_tokens_total", 0.0)
                g_sec = s.get("vllm:request_decode_time_seconds_sum", 0.0)
                kv_pct = s.get("vllm:kv_cache_usage_perc", 0.0) * 100.0
                running = int(s.get("vllm:num_requests_running", 0))
                waiting = int(s.get("vllm:num_requests_waiting", 0))

                p_tps = g_tps = "-"
                if url in prev:
                    pp_tok, pp_sec, pg_tok, pg_sec, _t = prev[url]
                    dp_sec = p_sec - pp_sec
                    dg_sec = g_sec - pg_sec
                    if dp_sec > 0.001 and p_tok >= pp_tok:
                        p_tps = f"{(p_tok - pp_tok) / dp_sec:.1f}"
                    if dg_sec > 0.001 and g_tok >= pg_tok:
                        g_tps = f"{(g_tok - pg_tok) / dg_sec:.1f}"
                prev[url] = (p_tok, p_sec, g_tok, g_sec, now)

                bans = s.get("antislop:ban_events_total")
                backtracks = s.get("antislop:backtracks_total")
                extra = f" | bans={int(bans)} backtracks={int(backtracks)}" if bans is not None else ""
                lines.append(
                    f"  {label:<10} run={running} wait={waiting} | KV={kv_pct:4.1f}% | "
                    f"prompt={p_tps} tok/s | gen={g_tps} tok/s{extra}"
                )

            print("\n".join(lines), flush=True)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()