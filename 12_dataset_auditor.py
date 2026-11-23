# -*- coding: utf-8 -*-
"""Dataset Entry Auditor — Notepad++-style viewer/editor + auto-scanner on localhost:5001.

Usage:
    python entry_auditor.py --input dataset_final.jsonl [--port 5001]
"""
import argparse
import json
import os
import re
import statistics
from collections import Counter
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

STOPWORDS = frozenset(
    "the a an and or but if in on at to for of with by from as is was are were be been "
    "being have has had do does did will would could should may might must can this that "
    "these those it its he she they we you i me him her them us my your his their our "
    "what which who whom whose where when why how all any both each few more most other "
    "some such no nor not only own same so than too very just about into over after".split()
)

LEAKED_JSON_RE = re.compile(r'\{\s*"(?:role|from|text|value)"\s*:')
SCRIPT_TAG_RE = re.compile(
    r'(\*\*\[SCENE\s+\d+\]\*\*|\*\*\(FADE\s+IN\)\*\*|\*\*(?:INT|EXT)\.\s+[A-Z]|\[[A-Z0-9 _]{2,20}\]:)',
)
WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def content_words(text):
    return {w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in STOPWORDS}


def analyze_entry(idx, entry):
    convs = entry.get("conversations", [])
    pairs = []
    i = 0
    while i < len(convs):
        if convs[i].get("from") == "human":
            q = convs[i].get("value", "")
            a = ""
            if i + 1 < len(convs) and convs[i + 1].get("from") == "gpt":
                a = convs[i + 1].get("value", "")
                i += 2
            else:
                i += 1
            pairs.append((q, a))
        else:
            i += 1

    pair_lengths = [len(q) + len(a) for q, a in pairs]
    total_chars = sum(len(c.get("value", "")) for c in convs)
    num_pairs = len(pairs)

    if pair_lengths:
        mean_len = round(statistics.mean(pair_lengths), 1)
        median_len = round(statistics.median(pair_lengths), 1)
        try:
            mode_len = statistics.mode(pair_lengths)
        except statistics.StatisticsError:
            mode_len = Counter(pair_lengths).most_common(1)[0][0]
    else:
        mean_len = median_len = mode_len = 0

    flags = []
    # 1. Leaked JSON
    for c in convs:
        if LEAKED_JSON_RE.search(c.get("value", "")):
            flags.append("leaked_json")
            break

    # 2. Duplicate human questions
    human_qs = [q.strip().lower() for q, _ in pairs if q.strip()]
    if len(human_qs) >= 3 and len(set(human_qs)) < len(human_qs) * 0.75:
        flags.append("duplicate_questions")

    # 3. Short fragment turns
    gpt_lens = [len(a.strip()) for _, a in pairs]
    if len(gpt_lens) >= 3 and sum(1 for l in gpt_lens if l < 80) >= len(gpt_lens) * 0.5:
        flags.append("short_fragment_turns")

    # 4. Script / transcript tags
    for _, a in pairs:
        if SCRIPT_TAG_RE.search(a):
            flags.append("script_tags")
            break

    # 5. Disjointed Q&A (multi-turn where answers share almost no content words with questions)
    if num_pairs >= 4:
        disjoint_count = 0
        for q, a in pairs:
            qw = content_words(q)
            aw = content_words(a)
            if len(qw) >= 3 and len(qw & aw) == 0:
                disjoint_count += 1
        if disjoint_count >= num_pairs * 0.6:
            flags.append("disjointed_qa")

    return {
        "index": idx,
        "num_pairs": num_pairs,
        "total_chars": total_chars,
        "mean_pair_len": mean_len,
        "median_pair_len": median_len,
        "mode_pair_len": mode_len,
        "flags": flags,
        "flagged": len(flags) > 0,
        "preview": (pairs[0][0][:80] if pairs else "").replace("\n", " "),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dataset Entry Auditor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #1e1e1e; color: #d4d4d4; height: 100vh; display: flex; flex-direction: column; }
  header { background: #2d2d30; padding: 8px 14px; border-bottom: 1px solid #3e3e42; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 15px; color: #fff; margin-right: 10px; }
  button, select, input { background: #3c3c3c; color: #e0e0e0; border: 1px solid #555; padding: 4px 9px; border-radius: 3px; font-size: 12px; cursor: pointer; }
  button:hover { background: #505050; }
  button.primary { background: #0e639c; border-color: #1177bb; color: #fff; }
  button.danger { background: #a1260d; border-color: #c53013; color: #fff; }
  button.success { background: #2d7d46; border-color: #389e58; color: #fff; }
  .main { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 340px; border-right: 1px solid #3e3e42; display: flex; flex-direction: column; background: #252526; }
  .list-controls { padding: 8px; border-bottom: 1px solid #3e3e42; display: flex; flex-direction: column; gap: 6px; }
  .entry-list { flex: 1; overflow-y: auto; }
  .entry-item { padding: 7px 10px; border-bottom: 1px solid #2d2d30; cursor: pointer; font-size: 12px; }
  .entry-item:hover { background: #2a2d2e; }
  .entry-item.active { background: #094771; color: #fff; }
  .entry-item.flagged { border-left: 4px solid #f44747; }
  .badge { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; margin-right: 4px; background: #f44747; color: #fff; }
  .workspace { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .stats-bar { background: #252526; padding: 10px 14px; border-bottom: 1px solid #3e3e42; display: flex; gap: 18px; flex-wrap: wrap; align-items: center; font-size: 12px; }
  .stat-box { background: #1e1e1e; padding: 5px 10px; border: 1px solid #3e3e42; border-radius: 3px; }
  .stat-box b { color: #4fc1ff; }
  .editor-wrap { flex: 1; display: flex; overflow: hidden; position: relative; }
  .npp-view { flex: 1; min-width: 120px; overflow: auto; padding: 12px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; line-height: 1.5; background: #1e1e1e; white-space: pre-wrap; word-break: break-word; }
  .resizer { width: 6px; background: #3e3e42; cursor: col-resize; flex-shrink: 0; transition: background 0.15s; }
  .resizer:hover, .resizer.dragging { background: #0e639c; }
  .npp-edit { flex: 1; min-width: 120px; background: #1b1b1b; color: #d4d4d4; border: none; padding: 12px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; line-height: 1.5; resize: none; outline: none; }
  .tok-key { color: #9cdcfe; }
  .tok-sys { color: #c586c0; font-weight: bold; }
  .tok-hum { color: #4ec9b0; font-weight: bold; }
  .tok-gpt { color: #ce9178; font-weight: bold; }
  .tok-act { color: #dcdcaa; font-style: italic; }
  .tok-tag { color: #f44747; font-weight: bold; }
</style>
</head>
<body>
<header>
  <h1>Dataset Entry Auditor</h1>
  <button class="primary" onclick="runScan()">Quick Scan All</button>
  <button class="primary" style="background:#6a1b9a;border-color:#8e24aa;" onclick="startLlmScan()">LLM Scan All</button>
  <select id="filterMode" onchange="renderList()">
    <option value="all">All Entries</option>
    <option value="flagged">Flagged Only</option>
    <option value="clean">Clean Only</option>
  </select>
  <select id="sortMode" onchange="renderList()">
    <option value="index">Original Order</option>
    <option value="len_asc">Shortest -> Longest</option>
    <option value="len_desc">Longest -> Shortest</option>
    <option value="mean_asc">Avg Pair Len (Low -> High)</option>
    <option value="mean_desc">Avg Pair Len (High -> Low)</option>
    <option value="pairs_desc">Most Q&A Pairs</option>
  </select>
  <input id="searchBox" type="text" placeholder="Search text or flag..." oninput="renderList()" style="width:180px;">
  <button class="danger" onclick="toggleFlagQueue()">Flag Queue (<span id="flagQueueCount">0</span>)</button>
  <span id="summaryText" style="font-size:12px;color:#aaa;margin-left:auto;">Loading...</span>
  <button class="success" onclick="exportClean()">Export Clean JSONL</button>
</header>
<div class="main">
  <div class="sidebar">
    <div class="entry-list" id="entryList"></div>
  </div>
  <div class="workspace">
    <div id="llmModal" style="display:none;background:#252526;border-bottom:2px solid #8e24aa;padding:10px 14px;font-size:12px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <b style="color:#ce93d8;">LLM Intelligent Scan</b>
        <span id="llmProgressText">Initializing...</span>
        <button class="danger" style="margin-left:auto;" onclick="stopLlmScan()">Stop Scan</button>
        <button onclick="document.getElementById('llmModal').style.display='none'">Hide</button>
      </div>
      <div style="background:#1e1e1e;height:8px;border-radius:4px;overflow:hidden;margin-bottom:6px;">
        <div id="llmProgressBar" style="background:#8e24aa;height:100%;width:0%;transition:width 0.2s;"></div>
      </div>
      <div id="llmLog" style="max-height:90px;overflow-y:auto;font-family:Consolas,monospace;font-size:11px;color:#ccc;"></div>
    </div>
    <div id="flagQueuePanel" style="display:none;background:#252526;border-bottom:2px solid #f44747;padding:10px 14px;font-size:12px;max-height:260px;overflow-y:auto;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
        <b style="color:#f44747;">Flag Queue</b>
        <button class="primary" style="background:#6a1b9a;border-color:#8e24aa;" onclick="llmDoubleScan()">LLM Double-Scan Flagged</button>
        <button class="success" onclick="saveFlaggedList()">Save Flagged List</button>
        <button class="danger" onclick="removeAllFlags()">Remove All Flags</button>
        <button style="margin-left:auto;" onclick="toggleFlagQueue()">Close</button>
      </div>
      <div id="flagQueueList"></div>
    </div>
    <div class="stats-bar" id="statsBar">
      <div class="stat-box">Entry: <b id="stIdx">-</b></div>
      <div class="stat-box">Q&A Pairs: <b id="stPairs">-</b></div>
      <div class="stat-box">Total Chars: <b id="stChars">-</b></div>
      <div class="stat-box">Mean Pair Len: <b id="stMean">-</b></div>
      <div class="stat-box">Median Pair Len: <b id="stMedian">-</b></div>
      <div class="stat-box">Mode Pair Len: <b id="stMode">-</b></div>
      <div id="stFlags"></div>
      <div style="margin-left:auto;display:flex;gap:6px;">
        <button class="primary" onclick="saveCurrent()">Save Edit</button>
        <button class="danger" onclick="deleteCurrent()">Delete Entry</button>
      </div>
    </div>
    <div class="editor-wrap" id="editorWrap">
      <div class="npp-view" id="nppView">Select an entry from the left...</div>
      <div class="resizer" id="editorResizer" title="Drag to resize"></div>
      <textarea class="npp-edit" id="nppEdit" spellcheck="false"></textarea>
  </div>
</div>
<script>
let metaList = [];
let currentIdx = null;

async function loadMeta() {
  const res = await fetch('/api/list');
  metaList = await res.json();
  renderList();
}

function renderList() {
  const filter = document.getElementById('filterMode').value;
  const sort = document.getElementById('sortMode').value;
  const q = document.getElementById('searchBox').value.toLowerCase();

  let items = metaList.filter(m => {
    if (filter === 'flagged' && !m.flagged) return false;
    if (filter === 'clean' && m.flagged) return false;
    if (q && !m.preview.toLowerCase().includes(q) && !m.flags.join(' ').toLowerCase().includes(q)) return false;
    return true;
  });

  if (sort === 'len_asc') items.sort((a,b) => a.total_chars - b.total_chars);
  else if (sort === 'len_desc') items.sort((a,b) => b.total_chars - a.total_chars);
  else if (sort === 'mean_asc') items.sort((a,b) => a.mean_pair_len - b.mean_pair_len);
  else if (sort === 'mean_desc') items.sort((a,b) => b.mean_pair_len - a.mean_pair_len);
  else if (sort === 'pairs_desc') items.sort((a,b) => b.num_pairs - a.num_pairs);
  else items.sort((a,b) => a.index - b.index);

  const flaggedCount = metaList.filter(m => m.flagged).length;
  document.getElementById('flagQueueCount').textContent = flaggedCount;
  document.getElementById('summaryText').textContent = `Showing ${items.length} / ${metaList.length} (${flaggedCount} flagged)`;
  renderFlagQueue();

  const listEl = document.getElementById('entryList');
  listEl.innerHTML = '';
  const frag = document.createDocumentFragment();
  for (const m of items) {
    const div = document.createElement('div');
    div.className = 'entry-item' + (m.flagged ? ' flagged' : '') + (m.index === currentIdx ? ' active' : '');
    div.onclick = () => selectEntry(m.index);
    const badges = m.flags.map(f => `<span class="badge">${f}</span>`).join('');
    div.innerHTML = `<div><b>#${m.index}</b> | ${m.num_pairs} pairs | ${m.total_chars}c | avg ${m.mean_pair_len}c</div>
                     <div style="margin-top:2px;">${badges}<span style="color:#999;">${escapeHtml(m.preview)}</span></div>`;
    frag.appendChild(div);
  }
  listEl.appendChild(frag);
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function highlightNpp(jsonStr) {
  let h = escapeHtml(jsonStr);
  h = h.replace(/"(conversations|from|value)"/g, '<span class="tok-key">"$1"</span>');
  h = h.replace(/"system"/g, '<span class="tok-sys">"system"</span>');
  h = h.replace(/"human"/g, '<span class="tok-hum">"human"</span>');
  h = h.replace(/"gpt"/g, '<span class="tok-gpt">"gpt"</span>');
  h = h.replace(/\*([^*\n]{1,120})\*/g, '<span class="tok-act">*$1*</span>');
  h = h.replace(/(\[[A-Z0-9 _]{2,25}\])/g, '<span class="tok-tag">$1</span>');
  return h;
}

async function selectEntry(idx) {
  currentIdx = idx;
  const res = await fetch('/api/entry?index=' + idx);
  const data = await res.json();
  const m = data.meta;
  document.getElementById('stIdx').textContent = '#' + m.index;
  document.getElementById('stPairs').textContent = m.num_pairs;
  document.getElementById('stChars').textContent = m.total_chars;
  document.getElementById('stMean').textContent = m.mean_pair_len;
  document.getElementById('stMedian').textContent = m.median_pair_len;
  document.getElementById('stMode').textContent = m.mode_pair_len;
  document.getElementById('stFlags').innerHTML = m.flags.map(f => `<span class="badge">${f}</span>`).join('');

  const pretty = JSON.stringify(data.entry, null, 2);
  document.getElementById('nppEdit').value = pretty;
  document.getElementById('nppView').innerHTML = highlightNpp(pretty);
  renderList();
}

document.getElementById('nppEdit').addEventListener('input', function() {
  document.getElementById('nppView').innerHTML = highlightNpp(this.value);
});

async function saveCurrent() {
  if (currentIdx === null) return;
  try {
    const parsed = JSON.parse(document.getElementById('nppEdit').value);
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({index: currentIdx, entry: parsed})
    });
    if (res.ok) { await loadMeta(); selectEntry(currentIdx); }
    else alert('Save failed');
  } catch (e) { alert('Invalid JSON: ' + e.message); }
}

async function deleteCurrent() {
  if (currentIdx === null || !confirm('Delete entry #' + currentIdx + '?')) return;
  await fetch('/api/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: currentIdx})
  });
  currentIdx = null;
  await loadMeta();
}

async function runScan() {
  await fetch('/api/quick_scan', {method: 'POST'});
  await loadMeta();
  document.getElementById('filterMode').value = 'flagged';
  renderList();
}

function toggleFlagQueue() {
  const el = document.getElementById('flagQueuePanel');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
  if (el.style.display === 'block') renderFlagQueue();
}

function renderFlagQueue() {
  const listEl = document.getElementById('flagQueueList');
  if (!listEl) return;
  const flagged = metaList.filter(m => m.flagged);
  if (!flagged.length) {
    listEl.innerHTML = '<span style="color:#888;">No entries currently flagged.</span>';
    return;
  }
  listEl.innerHTML = flagged.map(m => `
    <div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #333;">
      <b>#${m.index}</b>
      <span>${m.flags.map(f => `<span class="badge">${f}</span>`).join('')}</span>
      <span style="color:#aaa;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(m.preview)}</span>
      <button onclick="selectEntry(${m.index})">View</button>
      <button class="danger" onclick="unflagEntry(${m.index})">Unflag</button>
    </div>
  `).join('');
}

async function unflagEntry(idx) {
  await fetch('/api/unflag', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: idx})
  });
  await loadMeta();
}

async function removeAllFlags() {
  if (!confirm('Remove all flags from all entries?')) return;
  await fetch('/api/unflag_all', {method: 'POST'});
  await loadMeta();
}

async function llmDoubleScan() {
  const res = await fetch('/api/llm_scan/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({only_flagged: true})
  });
  const data = await res.json();
  if (!res.ok) { alert(data.error || 'Failed to start LLM double-scan'); return; }
  document.getElementById('llmModal').style.display = 'block';
  if (llmPollTimer) clearInterval(llmPollTimer);
  llmPollTimer = setInterval(pollLlmStatus, 600);
  pollLlmStatus();
}

async function saveFlaggedList() {
  const filename = prompt('Save flagged entries as:', 'dataset_flagged.jsonl');
  if (!filename) return;
  const res = await fetch('/api/export_flagged');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.jsonl') ? filename : filename + '.jsonl';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

let llmPollTimer = null;

async function startLlmScan() {
  const res = await fetch('/api/llm_scan/start', {method: 'POST'});
  const data = await res.json();
  if (!res.ok) { alert(data.error || 'Failed to start LLM scan'); return; }
  document.getElementById('llmModal').style.display = 'block';
  if (llmPollTimer) clearInterval(llmPollTimer);
  llmPollTimer = setInterval(pollLlmStatus, 600);
  pollLlmStatus();
}

async function stopLlmScan() {
  await fetch('/api/llm_scan/stop', {method: 'POST'});
}

async function pollLlmStatus() {
  const res = await fetch('/api/llm_scan/status');
  const st = await res.json();
  const pct = st.total ? Math.round((st.done / st.total) * 100) : 0;
  document.getElementById('llmProgressBar').style.width = pct + '%';
  document.getElementById('llmProgressText').textContent =
    `${st.done} / ${st.total} (${pct}%) — ${st.flagged} flagged` + (st.running ? ' [Running]' : ' [Finished]');
  document.getElementById('llmLog').innerHTML = st.logs.map(l => escapeHtml(l)).join('<br>');
  if (!st.running && llmPollTimer) {
    clearInterval(llmPollTimer);
    llmPollTimer = null;
    await loadMeta();
    document.getElementById('filterMode').value = 'flagged';
    renderList();
  }
}

async function exportClean() {
  const filename = prompt('Save clean dataset as:', 'dataset_clean.jsonl');
  if (!filename) return;
  const res = await fetch('/api/export_clean');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.jsonl') ? filename : filename + '.jsonl';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

(function initResizer() {
  const resizer = document.getElementById('editorResizer');
  const view = document.getElementById('nppView');
  const wrap = document.getElementById('editorWrap');
  let dragging = false;

  resizer.addEventListener('mousedown', (e) => {
    dragging = true;
    resizer.classList.add('dragging');
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = wrap.getBoundingClientRect();
    let newWidth = e.clientX - rect.left;
    const minW = 120;
    const maxW = rect.width - minW - resizer.offsetWidth;
    if (newWidth < minW) newWidth = minW;
    if (newWidth > maxW) newWidth = maxW;
    view.style.flex = 'none';
    view.style.width = newWidth + 'px';
  });

  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
  });
})();

loadMeta();
</script>
</body>
</html>
"""


class AuditorState:
    def __init__(self, input_path, llm_url, llm_model, llm_key):
        self.input_path = input_path
        self.llm_url = llm_url.rstrip("/")
        self.llm_model = llm_model
        self.llm_key = llm_key
        self.llm_flags_path = os.path.splitext(input_path)[0] + "_llm_flags.json"
        self.llm_flags = {}
        self.quick_scan_enabled = False
        self.unflagged = set()
        self.entries = []
        self.llm_status = {"running": False, "stop": False, "done": 0, "total": 0, "flagged": 0, "logs": []}
        self.load()
        if os.path.exists(self.llm_flags_path):
            try:
                with open(self.llm_flags_path, "r", encoding="utf-8") as f:
                    self.llm_flags = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                pass

    def save_llm_flags(self):
        with open(self.llm_flags_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.llm_flags.items()}, f, indent=2)

    def get_meta(self, idx):
        m = analyze_entry(idx, self.entries[idx])
        if not self.quick_scan_enabled or idx in self.unflagged:
            m["flags"] = []
            m["flagged"] = False
        if idx in self.llm_flags and idx not in self.unflagged:
            m["flags"].append("llm:" + self.llm_flags[idx])
            m["flagged"] = True
        return m

    def run_llm_scan_thread(self, only_flagged=False):
        if only_flagged:
            indices = [i for i in range(len(self.entries)) if self.get_meta(i)["flagged"]]
        else:
            indices = list(range(len(self.entries)))
        self.llm_status = {"running": True, "stop": False, "done": 0, "total": len(indices), "flagged": 0, "logs": []}
        for pos, idx in enumerate(indices):
            entry = self.entries[idx]
            if self.llm_status["stop"]:
                self.llm_status["logs"].append("Scan stopped by user.")
                break
            convs = entry.get("conversations", [])
            snippet = "\n".join(f"{c.get('from','?').upper()}: {str(c.get('value',''))[:350]}" for c in convs[:10])
            prompt = (
                "You are a dataset quality auditor. Inspect this conversation entry for corruption: "
                "leaked JSON/code, disjointed Q&A where answers ignore the questions, repeating/looping questions, "
                "or broken script fragments. Reply ONLY with valid JSON: "
                '{"corrupted": true/false, "reason": "short 2-4 word reason"}\n\n' + snippet[:3000]
            )
            try:
                req = urllib.request.Request(
                    f"{self.llm_url}/chat/completions",
                    data=json.dumps({
                        "model": self.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 120,
                    }).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        **({"Authorization": f"Bearer {self.llm_key}"} if self.llm_key else {}),
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", content, re.S)
                verdict = json.loads(m.group(0)) if m else {}
                if verdict.get("corrupted"):
                    reason = str(verdict.get("reason", "corrupted"))[:40]
                    self.unflagged.discard(idx)
                    self.llm_flags[idx] = reason
                    self.llm_status["flagged"] += 1
                    self.llm_status["logs"].append(f"[FLAGGED #{idx}] {reason}")
                    self.save_llm_flags()
                elif only_flagged:
                    self.unflagged.add(idx)
                    self.llm_flags.pop(idx, None)
                    self.llm_status["logs"].append(f"[CLEARED #{idx}] LLM verified clean")
                    self.save_llm_flags()
            except Exception as exc:
                self.llm_status["logs"].append(f"[ERR #{idx}] {exc}")
            self.llm_status["done"] = pos + 1
            if len(self.llm_status["logs"]) > 50:
                self.llm_status["logs"] = self.llm_status["logs"][-50:]
        self.llm_status["running"] = False

    def load(self):
        self.entries = []
        if not os.path.exists(self.input_path):
            return
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def save_all(self):
        tmp = self.input_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.input_path)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/list":
                meta = [state.get_meta(i) for i in range(len(state.entries))]
                self._send_json(meta)
            elif parsed.path == "/api/entry":
                qs = parse_qs(parsed.query)
                idx = int(qs.get("index", [0])[0])
                if 0 <= idx < len(state.entries):
                    self._send_json({
                        "meta": state.get_meta(idx),
                        "entry": state.entries[idx],
                    })
                else:
                    self._send_json({"error": "out of range"}, 404)
            elif parsed.path == "/api/llm_scan/status":
                self._send_json(state.llm_status)
            elif parsed.path == "/api/export_clean":
                clean = [e for i, e in enumerate(state.entries) if not state.get_meta(i)["flagged"]]
                body = ("\n".join(json.dumps(e, ensure_ascii=False) for e in clean) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/jsonl; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="dataset_clean.jsonl"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/export_flagged":
                flagged = [e for i, e in enumerate(state.entries) if state.get_meta(i)["flagged"]]
                body = ("\n".join(json.dumps(e, ensure_ascii=False) for e in flagged) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/jsonl; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="dataset_flagged.jsonl"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw)

            if self.path == "/api/save":
                idx = data["index"]
                if 0 <= idx < len(state.entries):
                    state.entries[idx] = data["entry"]
                    state.save_all()
                    self._send_json({"ok": True})
                else:
                    self._send_json({"error": "out of range"}, 404)
            elif self.path == "/api/delete":
                idx = data["index"]
                if 0 <= idx < len(state.entries):
                    state.entries.pop(idx)
                    state.save_all()
                    self._send_json({"ok": True})
                else:
                    self._send_json({"error": "out of range"}, 404)
            elif self.path == "/api/quick_scan":
                state.quick_scan_enabled = True
                state.unflagged.clear()
                self._send_json({"ok": True})
            elif self.path == "/api/unflag":
                idx = data["index"]
                state.unflagged.add(idx)
                state.llm_flags.pop(idx, None)
                state.save_llm_flags()
                self._send_json({"ok": True})
            elif self.path == "/api/unflag_all":
                state.quick_scan_enabled = False
                state.unflagged.clear()
                state.llm_flags.clear()
                state.save_llm_flags()
                self._send_json({"ok": True})
            elif self.path == "/api/llm_scan/start":
                if state.llm_status["running"]:
                    self._send_json({"error": "LLM scan already running"}, 400)
                else:
                    only_flagged = bool(data.get("only_flagged", False))
                    t = threading.Thread(target=state.run_llm_scan_thread, args=(only_flagged,), daemon=True)
                    t.start()
                    self._send_json({"ok": True})
            elif self.path == "/api/llm_scan/stop":
                state.llm_status["stop"] = True
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "not found"}, 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Dataset Entry Auditor on 10.0.0.179:5002")
    parser.add_argument("--input", "-i", default="dataset.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--port", "-p", type=int, default=5002, help="Port (default: 5002)")
    parser.add_argument("--llm-url", default="http://localhost:5001", help="OpenAI-compatible LLM API base URL")
    parser.add_argument("--llm-model", default="local-model", help="Model name for LLM scan")
    parser.add_argument("--llm-key", default="", help="Optional API key for LLM scan")
    args = parser.parse_args()

    state = AuditorState(args.input, args.llm_url, args.llm_model, args.llm_key)
    print(f"Loaded {len(state.entries)} entries from {args.input}")
    print(f"Serving Entry Auditor at http://10.0.0.179:{args.port} (LAN accessible)")
    server = ThreadingHTTPServer(("10.0.0.179", args.port), make_handler(state))
    server.serve_forever()


if __name__ == "__main__":
    main()
