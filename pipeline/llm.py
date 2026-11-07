# -*- coding: utf-8 -*-
"""The LLM client and everything around a model call.

The pipeline only ever uses one endpoint, /v1/chat/completions, so it runs
against llama-server, koboldcpp, vLLM, TabbyAPI, or any hosted OpenAI-compatible
API without changes.

Three things here are less obvious than the HTTP call itself:

  * `extract_json` — models wrap JSON in markdown fences, preambles and
    reasoning blocks. Parsing has to survive that.
  * `looks_like_refusal` — a refusal is a valid HTTP 200 with a polite
    paragraph in it. Steps treat refusals as "not done yet" rather than as
    results, so a rerun with a different model picks them up.
  * `parallel_map` — one worker means a plain loop, which keeps local
    single-slot servers happy and makes tracebacks readable.
"""
import json
import re
import sys
import time

import requests

# Refusal markers that are model-language independent enough to always check.
# Language packs add their own under `refusal_markers`.
BASE_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "cannot assist",
    "can't assist", "cannot help with", "can't help with", "i won't be able",
    "i will not be able", "i'm not able to", "i am not able to", "cannot fulfill",
    "can't fulfill", "cannot comply", "against my guidelines", "content policy",
    "as an ai", "as a language model",
)


class LLM:
    """Thin OpenAI-compatible chat client with retries and per-step override support."""

    def __init__(self, cfg, step_name=None):
        llm_cfg = cfg["llm"]
        override = llm_cfg.get(step_name, {}) if (step_name and isinstance(llm_cfg.get(step_name), dict)) else {}

        self.base_url = (override.get("base_url") or llm_cfg["base_url"]).rstrip("/")
        self.model = override.get("model") or llm_cfg.get("model", "local")
        self.api_key = override.get("api_key") or llm_cfg.get("api_key", "none")
        self.temperature = override.get("temperature", llm_cfg.get("temperature", 0.0))
        self.max_tokens = override.get("max_tokens", llm_cfg.get("max_tokens", 4096))
        self.retries = override.get("retries", llm_cfg.get("retries", 3))
        self.timeout = override.get("timeout", llm_cfg.get("timeout", 6000))
        self.step_name = step_name

    def chat(self, system, user, temperature=None, json_mode=False, max_tokens=None):
        """One call. Returns the message content as a string.

        Raises after `retries` failures, which the calling step treats as "retry
        this item on the next run" rather than as a reason to stop.
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            # Servers that do not know this field ignore it; servers that do
            # honour it save a whole class of parsing failures.
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - any failure is retryable here
                last_error = exc
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{self.retries}] {exc}; sleeping {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.retries} retries: {last_error}")

    def ping(self):
        """Check the server answers at all. Used by the smoke test."""
        reply = self.chat("You are a test fixture.", "Reply with the single word OK.",
                          temperature=0.0, max_tokens=16)
        return reply.strip()


def make_refusal_checker(extra_markers=()):
    """Build a refusal detector for a given language pack."""
    markers = tuple(BASE_REFUSAL_MARKERS) + tuple(m.lower() for m in extra_markers)

    def looks_like_refusal(text):
        # Only the head is checked: a refusal announces itself immediately,
        # while a legitimate answer may mention these phrases in passing later.
        head = (text or "").strip().lower()[:500]
        return any(marker in head for marker in markers)

    return looks_like_refusal


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)
_THINK_CLOSED = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_THINK_OPEN = re.compile(r"^<think(?:ing)?>", re.I)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _repair_json_string(text):
    """Attempt basic repairs on malformed or truncated JSON strings."""
    # Remove trailing commas inside objects and arrays
    text = re.sub(r",\s*([\]}])", r"\1", text)
    
    # Auto-close open strings, arrays, and objects if truncated
    in_string = False
    escape = False
    stack = []
    
    for char in text:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in "{[":
                stack.append('}' if char == '{' else ']')
            elif char in "}]":
                if stack and stack[-1] == char:
                    stack.pop()

    if in_string:
        text += '"'
    while stack:
        text += stack.pop()
        
    return text


def extract_json(text):
    """Pull a JSON value out of a model reply with heuristic salvage.

    Handles, in order: closed reasoning blocks, a reasoning block that got cut
    off by max_tokens, markdown fences, and finally a preamble before the JSON.
    """
    text = (text or "").strip()
    text = _THINK_CLOSED.sub("", text).strip()
    if _THINK_OPEN.match(text) and "</think" not in text.lower():
        # Unterminated reasoning block: the answer, if any, is after the last
        # opening brace.
        brace = text.rfind("{")
        text = text[brace:] if brace != -1 else text
    text = _FENCE.sub("", text).strip()

    # Fast path: clean JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract outermost balanced block { ... } or [ ... ]
    match = _JSON_RE.search(text)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            try:
                repaired = _repair_json_string(snippet)
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Salvage truncated array (e.g., scene boundaries)
    nums = re.findall(r"\b\d+\b", text)
    if nums and ("boundary" in text.lower() or "boundaries" in text.lower()):
        return {"boundaries": [int(n) for n in nums]}

    # Salvage truncated attribution payload
    if "characters" in text or "lines" in text:
        try:
            repaired = _repair_json_string(text[text.find("{"):])
            return json.loads(repaired)
        except Exception:
            pass

    # Salvage turns list from broken/truncated JSON transcripts:
    # Use non-greedy match between role tags to safely capture dialogue containing quotes
    if ('"role"' in text or "'role'" in text) and ('"text"' in text or "'text'" in text):
        pattern = re.compile(
            r'\{\s*["\']role["\']\s*:\s*["\'](char|user)["\']\s*,\s*["\']text["\']\s*:\s*["\'](.*?)["\']\s*(?:\}|$)',
            re.DOTALL | re.IGNORECASE
        )
        turn_matches = pattern.findall(text)
        if not turn_matches:
            # Try reverse key order: text first, role second
            rev_pattern = re.compile(
                r'\{\s*["\']text["\']\s*:\s*["\'](.*?)["\']\s*,\s*["\']role["\']\s*:\s*["\'](char|user)["\']\s*(?:\}|$)',
                re.DOTALL | re.IGNORECASE
            )
            rev_matches = rev_pattern.findall(text)
            if rev_matches:
                turn_matches = [(r, t) for (t, r) in rev_matches]

        if turn_matches:
            salvaged_turns = []
            for role, body in turn_matches:
                cleaned_body = (
                    body.replace('\\"', '"')
                        .replace("\\'", "'")
                        .replace('\\n', '\n')
                        .replace('\\t', '\t')
                        .strip()
                )
                if cleaned_body:
                    salvaged_turns.append({"role": role.lower(), "text": cleaned_body})
            if salvaged_turns:
                return {"turns": salvaged_turns}

    # Tagged block fallback: [user] or [char]
    tagged = re.findall(r'\[(char|user)\]\s*[:\-]?\s*(.*?)(?=\n\s*\[(?:char|user)\]|\Z)', text, re.DOTALL | re.IGNORECASE)
    if tagged:
        salvaged = [{"role": role.lower(), "text": body.strip()} for role, body in tagged if body.strip()]
        if salvaged:
            return {"turns": salvaged}

    raise ValueError(f"Unable to extract or repair JSON: {text[:200]}...")


def parallel_map(fn, items, workers):
    """Run fn over items, yielding (item, result, error) as results arrive.

    workers == 1 runs a plain sequential loop. That is not just an optimisation:
    a local single-slot server queues concurrent requests anyway, and sequential
    execution keeps the console log in a readable order.
    """
    items = list(items)
    if workers <= 1:
        for item in items:
            try:
                yield item, fn(item), None
            except Exception as exc:  # noqa: BLE001 - reported per item, never fatal
                yield item, None, exc
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                yield item, future.result(), None
            except Exception as exc:  # noqa: BLE001
                yield item, None, exc
