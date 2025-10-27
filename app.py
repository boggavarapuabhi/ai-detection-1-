import re, os
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
import httpx
from rapidfuzz import fuzz

app = FastAPI(title="AI Text & Plagiarism Detector", version="1.0.0")

# ---------- text utilities ----------
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
    return {
        "n_tokens": len(tokens),
        "ttr": ttr,
        "avg_word_len": avg_len,
        "stopword_ratio": stops,
        "caps_ratio": caps,
        "digit_ratio": digits
    }

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

# ---------- plagiarism via Bing Web Search ----------
async def bing_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    key = os.getenv("BING_SEARCH_KEY", "")
    if not key:
        return []
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"q": query, "count": top_k, "textDecorations": False, "textFormat": "Raw"}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(endpoint, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        return [{"url": it.get("url"), "title": it.get("name"), "snippet": it.get("snippet","")} for it in data.get("webPages", {}).get("value", [])]

def shingles(text: str, k: int = 12) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    return [" ".join(tokens[i:i+k]) for i in range(0, max(0, len(tokens)-k+1), max(1, k//2))]

async def plagiarism_hits_for(segment: str) -> List[Dict[str, Any]]:
    results = []
    for q in shingles(segment, k=12)[:4]:
        for h in await bing_search(q, 3):
            sim = fuzz.token_set_ratio(segment, h.get("snippet",""))
            if sim >= 70:
                results.append({"url": h["url"], "title": h["title"], "similarity": sim})
    # keep highest similarity per URL
    best = {}
    for r in results:
        u = r["url"]
        if u not in best or r["similarity"] > best[u]["similarity"]:
            best[u] = r
    return list(best.values())

# ---------- request/response ----------
class AnalyzeRequest(BaseModel):
    text: str
    ai_threshold: float = 0.65
    return_html: bool = True
    plagiarism: bool = True

class SpanResult(BaseModel):
    start: int
    end: int
    text: str
    ai_score: float
    ai_flag: bool
    plagiarism_hits: List[Dict[str, Any]] = []

class AnalyzeResponse(BaseModel):
    summary: Dict[str, Any]
    spans: List[SpanResult]
    html_preview: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = req.text
    sents = split_sentences(text)
    global_burst = burstiness(sents)
    spans = []
    cursor = 0

    for s in sents:
        idx = text.find(s, cursor)
        if idx < 0: idx = cursor
        start, end = idx, idx + len(s)
        cursor = end
        score = ai_likeness_score(s, global_burst)
        hits = await plagiarism_hits_for(s) if req.plagiarism else []

        spans.append(SpanResult(start=start, end=end, text=s,
            ai_score=round(score, 3), ai_flag=(score >= req.ai_threshold),
            plagiarism_hits=hits
        ))

    flagged = [sp for sp in spans if sp.ai_flag]
    plag = [sp for sp in spans if sp.plagiarism_hits]
    avg_score = round(sum(sp.ai_score for sp in spans)/len(spans), 3)

    # HTML highlighting
    html_parts = []
    last = 0
    for sp in spans:
        if last < sp.start:
            html_parts.append(text[last:sp.start])

        style = []
        if sp.ai_flag: style.append("background:rgba(255,0,0,0.2);")
        if sp.plagiarism_hits: style.append("text-decoration:underline wavy yellow;")

        if style:
            html_parts.append(f'<span style="{" ".join(style)}">{sp.text}</span>')
        else:
            html_parts.append(sp.text)

        last = sp.end

    if last < len(text):
        html_parts.append(text[last:])

    html = "<div style='font-family:Arial; line-height:1.5;'>" + "".join(html_parts) + "</div>"

    return AnalyzeResponse(
        summary={
            "avg_ai_score": avg_score,
            "ai_flagged_spans": len(flagged),
            "plagiarism_spans": len(plag),
            "total_spans": len(spans),
        },
        spans=spans,
        html_preview=html
    )
