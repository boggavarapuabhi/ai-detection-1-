import re
from typing import List, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="AI Text & Plagiarism Detector", version="1.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- text utils --------
_SENTENCE_RE = re.compile(r'(?us)([^.!?]+[.!?])')

def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.findall(text) or [text]
    return [p.strip() for p in parts if p.strip()]

def score_ai(text: str) -> int:
    # placeholder; plug your real model
    length = len(text.split())
    return max(0, min(100, 20 + length // 20))

def score_plagiarism(text: str) -> int:
    # placeholder; plug your rapidfuzz logic
    sents = split_sentences(text)
    repeated = len(sents) - len(set(sents))
    return max(0, min(100, repeated * 10))

# -------- models --------
class AnalyzeRequest(BaseModel):
    text: str

class SentenceOut(BaseModel):
    index: int
    text: str
    label: str             # "ai" | "plag" | "human"
    score: Optional[float] = None  # optional per-sentence score
    source: Optional[str] = None   # optional URL or match id

class AnalyzeJSON(BaseModel):
    summary: Dict[str, int]
    sentences: List[SentenceOut]

# -------- API that ChatGPT will call (JSON only, no HTML) --------
@app.post("/analyze", response_model=AnalyzeJSON)
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    sentences = split_sentences(text)

    ai_score = score_ai(text)
    plag_score = score_plagiarism(text)
    human_score = max(0, 100 - max(ai_score, plag_score))

    # demo labeling: long = AI-like, duplicate = plag
    seen = set()
    out: List[SentenceOut] = []
    for i, s in enumerate(sentences):
        is_plag = s in seen
        seen.add(s)
        is_ai = (len(s.split()) >= 18)

        label = "human"
        if is_plag:
            label = "plag"
        elif is_ai:
            label = "ai"

        out.append(SentenceOut(index=i, text=s, label=label))

    payload = {
        "summary": {
            "ai_score": ai_score,
            "plagiarism_score": plag_score,
            "human_score": human_score
        },
        "sentences": [o.dict() for o in out]
    }
    return JSONResponse(content=payload)

# -------- Optional: pretty HTML for YOUR WEBSITE ONLY --------
def _build_highlighted_html(sentences: List[str], labels: List[str]) -> str:
    from html import escape as _esc
    color_map = {"ai": "rgba(255,87,87,.25)", "plag": "rgba(255,193,7,.35)", "human": "rgba(76,175,80,.18)"}
    spans = []
    for i, s in enumerate(sentences):
        lab = labels[i]
        spans.append(f'<span style="background:{color_map[lab]};padding:2px 3px;border-radius:4px" data-index="{i}">{_esc(s)}</span>')
    body = " ".join(spans)
    return f"<div style='font-family:ui-sans-serif,system-ui;-webkit-font-smoothing:antialiased;line-height:1.6'>{body}</div>"

@app.post("/preview", response_class=HTMLResponse)
def preview(req: AnalyzeRequest):
    text = req.text.strip()
    sentences = split_sentences(text)
    labels = []
    seen = set()
    for s in sentences:
        lab = "human"
        if s in seen:
            lab = "plag"
        elif len(s.split()) >= 18:
            lab = "ai"
        labels.append(lab)
        seen.add(s)
    html = _build_highlighted_html(sentences, labels)
    return HTMLResponse(content=html)
