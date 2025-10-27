import re
from typing import List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="AI Text & Plagiarism Detector", version="1.5.0")

# CORS so your web UI or GPT Action can consume this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- simple text utils ----------
_SENTENCE_RE = re.compile(r'(?us)([^.!?]+[.!?])')

def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.findall(text) or [text]
    return [p.strip() for p in parts if p.strip()]

def score_ai(text: str) -> int:
    # placeholder. plug your real logic here.
    length = len(text.split())
    return max(0, min(100, 20 + length // 20))

def score_plagiarism(text: str) -> int:
    # placeholder. plug your real logic here.
    # treat repeated sentences as "plag-like" just to demo colors
    sents = split_sentences(text)
    repeated = len(sents) - len(set(sents))
    return max(0, min(100, repeated * 10))

def build_highlighted_html(sentences: List[str], ai_mask: List[bool], plag_mask: List[bool]) -> str:
    from html import escape as _esc

    colored = []
    for i, s in enumerate(sentences):
        cls = "sent human"
        if plag_mask[i]:
            cls = "sent plag"
        elif ai_mask[i]:
            cls = "sent ai"
        colored.append(f'<span class="{cls}" data-index="{i}">{_esc(s)}</span>')

    body = " ".join(colored)

    styles = """
    <style>
      .report { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; line-height:1.6 }
      .summary { display:flex; gap:12px; margin-bottom:12px; flex-wrap:wrap }
      .chip{padding:6px 10px;border-radius:999px;border:1px solid rgba(0,0,0,.08);font-size:13px}
      .sent { padding:2px 3px; border-radius:4px }
      .sent.ai   { background: rgba(255, 87, 87, .25); }     /* red for AI-likely */
      .sent.plag { background: rgba(255, 193, 7, .35); }     /* amber for plag */
      .sent.human{ background: rgba(76, 175, 80, .18); }     /* green for human */
    </style>
    """

    header = """
    <div class="summary">
      <span class="chip" id="ai_chip"></span>
      <span class="chip" id="plag_chip"></span>
      <span class="chip" id="human_chip"></span>
    </div>
    """

    script = """
    <script>
      if (typeof window !== 'undefined' && window.__summary__) {
        const {ai, plag, human} = window.__summary__;
        document.getElementById('ai_chip').textContent = `AI-generated: ${ai}%`;
        document.getElementById('plag_chip').textContent = `Plagiarism: ${plag}%`;
        document.getElementById('human_chip').textContent = `Human: ${human}%`;
      }
    </script>
    """

    return f'{styles}<div class="report">{header}<div class="body">{body}</div>{script}</div>'

# ---------- request/response shapes ----------
class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    summary: Dict[str, int]
    highlighted_html: str

# ---------- main endpoints ----------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    sentences = split_sentences(text)

    ai_score = score_ai(text)
    plag_score = score_plagiarism(text)
    human_score = max(0, 100 - max(ai_score, plag_score))

    # demo masks: long sentences count as AI-like, duplicates as plag-like
    ai_mask = [len(s.split()) >= 18 for s in sentences]
    seen = set()
    plag_mask = []
    for s in sentences:
        if s in seen:
            plag_mask.append(True)
        else:
            plag_mask.append(False)
            seen.add(s)

    html_block = build_highlighted_html(sentences, ai_mask, plag_mask)

    payload = {
        "summary": {
            "ai_score": ai_score,
            "plagiarism_score": plag_score,
            "human_score": human_score
        },
        "highlighted_html": html_block
    }
    return JSONResponse(content=payload)

# optional quick preview as a full HTML page
@app.post("/preview", response_class=HTMLResponse)
def preview(req: AnalyzeRequest):
    data = analyze(req).body.decode("utf-8")
    import json
    obj = json.loads(data)
    html = f"""
      <html><head><meta charset="utf-8"><title>Detector Preview</title></head>
      <body>
        <script>window.__summary__={{ai:{obj['summary']['ai_score']},plag:{obj['summary']['plagiarism_score']},human:{obj['summary']['human_score']}}}</script>
        {obj['highlighted_html']}
      </body></html>
    """
    return HTMLResponse(content=html)
