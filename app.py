import os, re, io, html
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx
from rapidfuzz import fuzz
from html import escape
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

app = FastAPI(title="AI Text & Plagiarism Detector", version="1.4.0")

# ---------- text utils ----------
_SENTENCE_RE = re.compile(r'(?us)([^.!?]+[.!?])', re.MULTILINE)
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by could did do does doing down during each few for from
further had has have having he her here hers himself him himself his how i if in into
is it its itself just me more most my myself no nor not of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was
we were what when where which while who whom why will with you your yours yourself yourselves
""".split())

def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.findall(text.strip())
    if not parts:
        return [text.strip()] if text.strip() else []
    return [p.strip() for p in parts]

def word_stats(s: str):
    tokens = re.findall(r"[A-Za-z']+", s.lower())
    if not tokens:
        return {"n_tokens": 0, "ttr": 0.0, "avg_word_len": 0.0, "stopword_ratio": 0.0, "caps_ratio": 0.0, "digit_ratio": 0.0}
    unique = len(set(tokens))
    ttr = unique / len(tokens)
    avg_len = sum(len(t) for t in tokens)/len(tokens)
    stops = sum(1 for t in tokens if t in STOPWORDS) / len(tokens)
    caps = sum(1 for ch in s if ch.isupper()) / max(1, len(s))
    digits = sum(1 for ch in s if ch.isdigit()) / max(1, len(s))
    return {"n_tokens": len(tokens), "ttr": ttr, "avg_word_len": avg_len, "stopword_ratio": stops, "caps_ratio": caps, "digit_ratio": digits}

def burstiness(sentences: List[str]) -> float:
    if not sentences: return 0.0
    lens = [len(s.split()) for s in sentences if s.strip()]
    if not lens: return 0.0
    mean = sum(lens)/len(lens)
    var = sum((x-mean)**2 for x in lens)/len(lens)
    return var / (mean**2 + 1e-6)

def ai_likeness_score(s: str, local_burst: float) -> float:
    st = word_stats(s)
    if st["n_tokens"] == 0: return 0.0
    f_ttr = 1.0 - min(1.0, st["ttr"])
    f_burst = 1.0 - max(0.0, min(1.0, local_burst))
    f_stop = 1.0 - abs(st["stopword_ratio"] - 0.45)*2.0; f_stop = max(0.0, min(1.0, f_stop))
    f_avglen = 1.0 - abs(st["avg_word_len"] - 5.0)/5.0;  f_avglen = max(0.0, min(1.0, f_avglen))
    f_caps = 1.0 - min(1.0, st["caps_ratio"]*10)
    f_digit = 1.0 - min(1.0, st["digit_ratio"]*10)
    w = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
    feats = [f_ttr, f_burst, f_stop, f_avglen, f_caps, f_digit]
    score = sum(wi*fi for wi,fi in zip(w,feats))
    return max(0.0, min(1.0, score))

# ---------- plagiarism (optional) ----------
async def bing_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    key = os.getenv("BING_SEARCH_KEY", "")
    if not key:
        return []
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"q": query, "count": top_k}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(endpoint, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        return [{"url": it.get("url"), "title": it.get("name"), "snippet": it.get("snippet","")} for it in data.get("webPages", {}).get("value", [])]

def make_shingles(text: str, k: int = 12):
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    if len(tokens) < k: return []
    step = max(1, k//2)
    return [" ".join(tokens[i:i+k]) for i in range(0, len(tokens)-k+1, step)]

async def plagiarism_hits_for(segment: str) -> List[Dict[str, Any]]:
    results = []
    for q in make_shingles(segment, 12)[:3]:
        for h in await bing_search(q, 3):
            sim = fuzz.token_set_ratio(segment, h.get("snippet",""))
            if sim >= 70:
                results.append({"url": h["url"], "title": h["title"], "similarity": sim})
    best = {}
    for r in results:
        u = r["url"]
        if u not in best or r["similarity"] > best[u]["similarity"]:
            best[u] = r
    return list(best.values())

# ---------- models ----------
class AnalyzeRequest(BaseModel):
    text: str
    ai_threshold: float = 0.65
    return_html: bool = True
    plagiarism: bool = True
    show_only_flagged: bool = False
    show_score_badges: bool = True

class SpanResult(BaseModel):
    index: int
    start: int
    end: int
    text: str
    ai_score: float
    ai_label: str
    ai_flag: bool
    plagiarism_hits: List[Dict[str, Any]] = []

class AnalyzeResponse(BaseModel):
    summary: Dict[str, Any]
    spans: List[SpanResult]
    html_preview: str = ""
    markdown_summary: str = ""   # NEW: always-readable summary

# ---------- helpers ----------
def ai_label_from_score(score: float, threshold: float) -> str:
    if score < threshold: return "human"
    if score >= 0.85: return "ai_high"
    if score >= 0.75: return "ai_med"
    return "ai_low"

def build_markdown_summary(text: str, spans: List[SpanResult], summary: Dict[str,Any], only_flagged: bool) -> str:
    # Emoji legend
    legend = "🟩 Human-like  |  🟧 AI (low)  |  🟨 AI (med)  |  🟥 AI (high)  |  🔶 Underline = Plagiarism"
    lines = []
    lines.append("### Summary")
    lines.append(f"- **Avg AI score:** `{summary['avg_ai_score']}`")
    lines.append(f"- **AI-flagged sentences:** `{summary['ai_flagged_spans']}` / `{summary['total_spans']}`")
    lines.append(f"- **Plagiarism sentences:** `{summary['plagiarism_spans']}`")
    lines.append("")
    lines.append("### Legend")
    lines.append(legend)
    lines.append("")
    lines.append("### Sentences")
    target = [sp for sp in spans if (sp.ai_flag or sp.plagiarism_hits)] if only_flagged else spans
    if not target:
        lines.append("_No sentences to show._")
    else:
        for sp in target:
            color = {"human":"🟩","ai_low":"🟧","ai_med":"🟨","ai_high":"🟥"}[sp.ai_label]
            plag = " • 🔶 plagiarism" if sp.plagiarism_hits else ""
            short = sp.text.strip().replace("\n"," ")
            if len(short)>220: short=short[:217]+"…"
            lines.append(f"- **[{sp.index}]** {color} `{sp.ai_score:.2f}`{plag} — {short}")
    lines.append("")
    lines.append("> _Note: Signals are probabilistic — not definitive proof._")
    return "\n".join(lines)

def render_html_block(full_text, spans, avg_score, ai_count, plag_count, show_only_flagged, show_score_badges):
    def risk(score, label):
        if label=="human": return "human"
        if score>=0.85: return "high"
        if score>=0.75: return "med"
        return "low"

    # Build inside page
    parts=[]
    if show_only_flagged:
        flagged=[s for s in spans if s.ai_flag or s.plagiarism_hits]
        if not flagged:
            parts.append("<div class='muted'>No flagged sentences.</div>")
        for sp in flagged:
            txt=escape(sp.text)
            cls=f"sent {risk(sp.ai_score, sp.ai_label)}"
            deco=" plag" if sp.plagiarism_hits else ""
            badge=f"<span class='badge'>{sp.ai_score:.2f}</span>" if show_score_badges else ""
            parts.append(f"<div class='flag'><span class='{cls}{deco}' title='AI {sp.ai_score:.2f}; Plag {len(sp.plagiarism_hits)}'>{txt}</span>{badge}</div>")
    else:
        last=0
        # spans already carry start/end; we just join in order
        for sp in spans:
            if last<sp.start: parts.append(escape(full_text[last:sp.start]))
            txt=escape(sp.text)
            cls=f"sent {risk(sp.ai_score, sp.ai_label)}"
            deco=" plag" if sp.plagiarism_hits else ""
            badge=f"<span class='badge'>{sp.ai_score:.2f}</span>" if show_score_badges else ""
            parts.append(f"<span class='{cls}{deco}' title='AI {sp.ai_score:.2f}; Plag {len(sp.plagiarism_hits)}'>{txt}</span>{badge}")
            last=sp.end
        if last<len(full_text): parts.append(escape(full_text[last:]))

    inner = f"""
<style>
  .report {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; line-height:1.6; }}
  .row {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
  .h1 {{ font-size:18px; font-weight:700; }}
  .muted {{ color:#6b7280; font-size:13px; }}
  .chip {{ font-size:12px; padding:3px 8px; border-radius:999px; background:rgba(0,0,0,.05); border:1px solid rgba(0,0,0,.08); }}
  .card {{ border:1px solid rgba(0,0,0,.08); border-radius:12px; padding:12px; background:rgba(255,255,255,.9); }}
  @media (prefers-color-scheme: dark) {{
    .card {{ background:rgba(30,30,30,.7); border-color:rgba(255,255,255,.12); }}
    .muted {{ color:#9ca3af; }}
    .chip {{ background:rgba(255,255,255,.06); border-color:rgba(255,255,255,.12); }}
  }}
  .sent {{ padding:0 2px; border-radius:6px; }}
  .sent.human {{ background:#ecfdf5; border:1px solid #10b98133; }}
  .sent.low   {{ background:#fff4e6; }}
  .sent.med   {{ background:#ffe4e6; }}
  .sent.high  {{ background:#fee2e2; border:1px solid #ef444433; }}
  .plag {{ text-decoration: underline wavy; text-decoration-color:#c9a400; }}
  .badge {{ display:inline-block; margin-left:6px; font-size:11px; padding:2px 6px; border-radius:6px; background:#1118270d; border:1px solid #1118271a; }}
  .flag {{ margin:6px 0; }}
</style>

<div class="report">
  <div class="row">
    <div class="h1">Highlighted Results</div>
    <span class="chip">AI-flagged: {ai_count}</span>
    <span class="chip">Plagiarism: {plag_count}</span>
    <span class="chip">Avg score: {avg_score:.2f}</span>
    <span class="chip">{'Only flagged' if show_only_flagged else 'All sentences'}</span>
  </div>
  <div class="card">{''.join(parts)}</div>
  <div class="muted">Note: Signals are probabilistic — not definitive proof.</div>
</div>
""".strip()

    # Important: wrap with an iframe so GPT renders even when it would otherwise echo
    srcdoc = html.escape(inner, quote=True)
    iframe = f"<iframe style='width:100%;height:420px;border:none;border-radius:8px' srcdoc='{srcdoc}'></iframe>"
    return iframe

# ---------- core pipeline ----------
@app.get("/healthz")
def healthz(): return {"ok": True}

async def _analyze(req: AnalyzeRequest):
    text = req.text
    sents = split_sentences(text)
    b = burstiness(sents)
    spans: List[SpanResult] = []
    cursor = 0
    for i, s in enumerate(sents, start=1):
        idx = text.find(s, cursor)
        if idx < 0: idx = cursor
        start, end = idx, idx + len(s)
        cursor = end
        score = ai_likeness_score(s, b)
        label = ai_label_from_score(score, req.ai_threshold)
        hits = await plagiarism_hits_for(s) if req.plagiarism else []
        spans.append(SpanResult(index=i, start=start, end=end, text=s,
                                ai_score=round(score,3), ai_label=label,
                                ai_flag=(score >= req.ai_threshold),
                                plagiarism_hits=hits))
    n = len(spans) or 1
    ai_count = sum(1 for sp in spans if sp.ai_flag)
    plag_count = sum(1 for sp in spans if sp.plagiarism_hits)
    avg_score = round(sum(sp.ai_score for sp in spans)/n, 3)

    summary = {
        "avg_ai_score": avg_score,
        "ai_flagged_spans": ai_count,
        "plagiarism_spans": plag_count,
        "total_spans": n,
        "engine": "heuristic-v1",
        "plagiarism_engine": "bing-web" if os.getenv("BING_SEARCH_KEY") else "disabled"
    }

    md = build_markdown_summary(text, spans, summary, req.show_only_flagged)
    html_block = render_html_block(text, spans, avg_score, ai_count, plag_count,
                                   req.show_only_flagged, req.show_score_badges) if req.return_html else ""

    return summary, spans, html_block, md

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    summary, spans, html_block, md = await _analyze(req)
    return AnalyzeResponse(summary=summary, spans=spans, html_preview=html_block, markdown_summary=md)

# ---------- PDF report ----------
@app.post("/report")
async def report(req: AnalyzeRequest):
    summary, spans, _, _ = await _analyze(req)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x, y = inch*0.75, height - inch*0.75

    def line(txt, size=11, lead=14, bold=False):
        nonlocal y
        if y < inch: 
            c.showPage(); y = height - inch*0.75
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, txt[:110])
        y -= lead

    line("AI Text & Plagiarism Detector — Report", 14, 18, True)
    line("Note: Signals are probabilistic — not definitive.", 9, 12)
    s = summary
    line(""); line("Summary", 12, 16, True)
    line(f"Average AI score: {s['avg_ai_score']}")
    line(f"AI-flagged spans: {s['ai_flagged_spans']}/{s['total_spans']}")
    line(f"Plagiarism spans: {s['plagiarism_spans']}")

    flagged = [sp for sp in spans if sp.ai_flag or sp.plagiarism_hits]
    flagged.sort(key=lambda sp: sp.ai_score, reverse=True)
    line(""); line("Flagged (Top 10)", 12, 16, True)
    if not flagged: line("None")
    else:
        for sp in flagged[:10]:
            line(f"[{sp.index}] {sp.ai_score:.2f} — {sp.text[:95]}")

    sources = {}
    for sp in spans:
        for h in sp.plagiarism_hits:
            u = h.get("url",""); 
            if not u: continue
            if u not in sources or h.get("similarity",0) > sources[u].get("similarity",0):
                sources[u] = h
    line(""); line("Plagiarism sources", 12, 16, True)
    if not sources: line("None")
    else:
        for u, h in list(sources.items())[:10]:
            line(f"{h.get('title','Source')} — Sim {h.get('similarity',0)}%")
            line(u)

    c.showPage(); c.save(); buffer.seek(0)
    headers = {"Content-Disposition": "inline; filename=ai-report.pdf"}
    return StreamingResponse(buffer, media_type="application/pdf", headers=headers)
