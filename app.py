import re
import asyncio
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# External libs
import httpx
from rapidfuzz import fuzz

app = FastAPI(title="AI Text & Plagiarism Detector", version="1.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utils
# -----------------------------
_SENTENCE_RE = re.compile(r'(?us)([^.!?]+[.!?])')
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.findall(text) or [text]
    return [p.strip() for p in parts if p.strip()]

def normalize_text(text: str) -> str:
    # strip tags, collapse spaces
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text

def ai_score_simple(text: str) -> int:
    # Placeholder heuristic. Replace with your model later.
    # Longer, generic sentences push score up.
    sents = split_sentences(text)
    long_count = sum(len(s.split()) >= 18 for s in sents)
    score = min(100, 15 + long_count * 12)
    return max(0, score)

# -----------------------------
# Plag detection
# -----------------------------
async def fetch_url_text(url: str, client: httpx.AsyncClient) -> str:
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        return normalize_text(r.text)
    except Exception:
        return ""

async def build_corpus(source_texts: List[str], source_urls: List[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (source_id, clean_text).
    source_id is either 'text:0', 'text:1', ... or the URL.
    """
    corpus: List[Tuple[str, str]] = []
    # Provided raw texts
    for i, t in enumerate(source_texts or []):
        if t and t.strip():
            corpus.append((f"text:{i}", normalize_text(t)))

    # URL fetches
    if source_urls:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [fetch_url_text(u, client) for u in source_urls]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        for u, body in zip(source_urls, results):
            if body:
                corpus.append((u, body))

    return corpus

def best_match_for_sentence(sent: str, corp_text: str) -> Tuple[int, str]:
    """
    Find best RapidFuzz similarity for a sentence within a corpus text by
    comparing against corpus sentences. Returns (best_score, best_snippet).
    """
    best_score = 0
    best_snip = ""
    # Compare with sentence-level windows from corpus
    corp_sents = split_sentences(corp_text)
    for cs in corp_sents:
        # mix of token & partial to handle paraphrase vs exact
        s1 = fuzz.token_set_ratio(sent, cs)
        s2 = fuzz.partial_ratio(sent, cs)
        score = max(s1, s2)
        if score > best_score:
            best_score = score
            best_snip = cs
    return best_score, best_snip

# -----------------------------
# API models
# -----------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Plain text to analyze")
    source_texts: Optional[List[str]] = Field(default=None, description="Optional reference texts to check against")
    source_urls: Optional[List[str]]  = Field(default=None, description="Optional web pages to fetch and compare")
    plag_threshold: int = Field(default=85, ge=50, le=100, description="Similarity threshold to mark as plagiarism")

class SentenceOut(BaseModel):
    index: int
    text: str
    label: str                 # "ai" | "plag" | "human"
    ai_like: Optional[bool] = None
    plag_score: Optional[int] = None
    source: Optional[str] = None
    source_snippet: Optional[str] = None

class AnalyzeJSON(BaseModel):
    summary: Dict[str, int]
    sentences: List[SentenceOut]

# -----------------------------
# Core analyzer
# -----------------------------
async def analyze_core(req: AnalyzeRequest) -> AnalyzeJSON:
    text = req.text.strip()
    sentences = split_sentences(text)

    # Scores
    ai_score = ai_score_simple(text)
    plag_threshold = req.plag_threshold

    # Build external corpus
    corpus = await build_corpus(req.source_texts or [], req.source_urls or [])

    # Label each sentence
    out: List[SentenceOut] = []
    seen = set()
    for i, s in enumerate(sentences):
        # Duplicate in the same essay -> weak plagiarism signal
        is_dupe = s in seen
        seen.add(s)

        best_plag_score = 0
        best_source = None
        best_snippet = None

        # Check against external corpus
        for src_id, body in corpus:
            score, snip = best_match_for_sentence(s, body)
            if score > best_plag_score:
                best_plag_score = score
                best_source = src_id
                best_snippet = snip

        # Decide label priority: plagiarism > AI-like > human
        label = "human"
        ai_like = (len(s.split()) >= 18) or (ai_score >= 70)

        if is_dupe:
            # self-duplication gets lower score; treat as mild plag if long
            best_plag_score = max(best_plag_score, 70 if len(s.split()) > 8 else 0)

        if best_plag_score >= plag_threshold:
            label = "plag"
        elif ai_like:
            label = "ai"

        out.append(SentenceOut(
            index=i,
            text=s,
            label=label,
            ai_like=ai_like,
            plag_score=best_plag_score if best_plag_score > 0 else None,
            source=best_source,
            source_snippet=best_snippet
        ))

    # Summary
    plag_sent = sum(1 for o in out if o.label == "plag")
    ai_sent = sum(1 for o in out if o.label == "ai")
    total = max(1, len(out))

    summary = {
        "ai_score": ai_score,
        "plagiarism_score": min(100, int(100 * plag_sent / total)),  # simple ratio
        "human_score": max(0, 100 - max(ai_score, int(100 * plag_sent / total)))
    }

    return AnalyzeJSON(summary=summary, sentences=out)

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/analyze", response_model=AnalyzeJSON)
async def analyze(req: AnalyzeRequest):
    result = await analyze_core(req)
    return JSONResponse(content=result.model_dump())

# Optional HTML preview for YOUR website (not for ChatGPT)
def _build_html(sentences: List[SentenceOut]) -> str:
    from html import escape as _esc
    colors = {"ai": "rgba(255,87,87,.25)", "plag": "rgba(255,193,7,.35)", "human": "rgba(76,175,80,.18)"}
    spans = []
    for s in sentences:
        spans.append(
            f'<span title="{_esc((s.source or "") + (" | score " + str(s.plag_score) if s.plag_score else ""))}" '
            f'style="background:{colors[s.label]};padding:2px 3px;border-radius:4px" '
            f'data-index="{s.index}">{_esc(s.text)}</span>'
        )
    return "<div style='font-family:ui-sans-serif,system-ui;line-height:1.6'>" + " ".join(spans) + "</div>"

@app.post("/preview", response_class=HTMLResponse)
async def preview(req: AnalyzeRequest):
    result = await analyze_core(req)
    html = _build_html(result.sentences)
    return HTMLResponse(content=html)
