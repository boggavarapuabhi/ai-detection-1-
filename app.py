import re
import io
import asyncio
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
from rapidfuzz import fuzz

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

app = FastAPI(title="AI Detector PRO", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------- Utilities --------------
_SENTENCE_RE = re.compile(r'(?us)([^.!?]+[.!?])')
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def split_sentences(text: str) -> List[str]:
    sents = _SENTENCE_RE.findall(text) or [text]
    return [s.strip() for s in sents if s.strip()]

def normalize_text(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()

def stitch(sents: List[str]) -> str:
    return " ".join(s.strip() for s in sents if s.strip())

def ai_score(text: str) -> int:
    sents = split_sentences(text)
    long_count = sum(len(x.split()) >= 18 for x in sents)
    return min(100, 15 + long_count * 12)

async def fetch_text(url: str, client: httpx.AsyncClient) -> str:
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        return normalize_text(r.text)
    except:
        return ""

async def build_corpus(texts: List[str], urls: List[str]) -> List[Tuple[str,str]]:
    corpus = [(f"text:{i}", normalize_text(t)) for i,t in enumerate(texts or []) if t.strip()]
    if urls:
        async with httpx.AsyncClient(follow_redirects=True) as c:
            res = await asyncio.gather(*[fetch_text(u, c) for u in urls])
        for u,body in zip(urls, res):
            if body:
                corpus.append((u, body))
    return corpus

def best_match(sent: str, corp_text: str):
    best, snip = 0, ""
    for cs in split_sentences(corp_text):
        sc = max(fuzz.token_set_ratio(sent, cs), fuzz.partial_ratio(sent, cs))
        if sc > best:
            best, snip = sc, cs
    return best, snip

def reason_ai(s: str):
    words = s.split()
    long = len(words)>=22
    low_var = len(set(w.lower().strip(",.!?")) for w in words) < len(words)*0.65
    passive = any(x in s.lower() for x in ["is being","was","were","has been"])
    reasons=[]
    if long: reasons.append("very long")
    if low_var: reasons.append("low variety")
    if passive: reasons.append("passive tone")
    return ", ".join(reasons) if reasons else "generic phrasing"

def fix_ai(): return "Add personal detail & varied verbs."
def fix_plag(): return "Rewrite in your own words & cite sources."

# ---------- API Models ----------
class AnalyzeIn(BaseModel):
    text: str
    source_texts: Optional[List[str]]=None
    source_urls: Optional[List[str]]=None
    plag_threshold: int=85

class SentenceOut(BaseModel):
    index:int
    text:str
    label:str
    confidence:str
    plag_score:Optional[int]=None
    source:Optional[str]=None
    snippet:Optional[str]=None
    reason:Optional[str]=None
    fix:Optional[str]=None

class AnalyzeOut(BaseModel):
    summary:dict
    counts:dict
    sentences:List[SentenceOut]
    markdown_report:str

# ------------ Detection Core ------------
async def run_analysis(req:AnalyzeIn)->AnalyzeOut:
    text=req.text.strip()
    sents=split_sentences(text)
    ai_over=ai_score(text)
    corp=await build_corpus(req.source_texts or [], req.source_urls or [])

    out=[]
    seen=set()
    ai_n=pl_n=0

    for i,s in enumerate(sents):
        is_dupe = s in seen
        seen.add(s)

        best,src,snip=0,None,None
        for u,b in corp:
            sc,mp = best_match(s,b)
            if sc>best: best,src,snip=sc,u,mp

        label="human"
        if best>=req.plag_threshold: label="plag"
        elif len(s.split())>=18 or ai_over>=70: label="ai"

        if label=="ai": ai_n+=1
        if label=="plag": pl_n+=1

        conf="High" if (label=="plag" and best>=92) or (label=="ai" and ai_over>=82) else "Medium" if label!="human" else "Low"
        
        out.append(SentenceOut(
            index=i,
            text=s,
            label=label,
            confidence=conf,
            plag_score=best if best else None,
            source=src,
            snippet=snip,
            reason=reason_ai(s) if label=="ai" else ("high similarity" if label=="plag" else None),
            fix=fix_ai() if label=="ai" else (fix_plag() if label=="plag" else None)
        ))

    total=len(out)
    summary={
        "ai_score":ai_over,
        "plagiarism_score":min(100,int(100*pl_n/max(1,total))),
        "human_score":max(0,100-max(ai_over,summary["plagiarism_score"] if pl_n else 0))
    }

    counts={"total":total,"ai":ai_n,"plag":pl_n,"human":total-ai_n-pl_n}

    # Markdown Summary for ChatGPT
    mr=["## Report"]
    mr.append(f"- **AI-likeness:** {summary['ai_score']}%")
    mr.append(f"- **Plagiarism:** {summary['plagiarism_score']}%")
    mr.append(f"- **Human-likeness:** {summary['human_score']}%")
    mr.append("✅ Use these results as guidance, not proof.")

    markdown_report="\n".join(mr)
    return AnalyzeOut(summary=summary,counts=counts,sentences=out,markdown_report=markdown_report)

# -------------- API Endpoints --------------
@app.post("/analyze",response_model=AnalyzeOut)
async def analyze(req:AnalyzeIn):
    return JSONResponse((await run_analysis(req)).dict())

class RewriteIn(BaseModel):
    sentence:str
    style:str="human"

class RewriteOut(BaseModel):
    original:str
    rewritten:str
    style:str
    tip:str

@app.post("/rewrite",response_model=RewriteOut)
def rewrite(req:RewriteIn):
    s=req.sentence
    if req.style=="human": s=s+" (more personal insight)"
    return {
        "original":req.sentence,
        "rewritten":s,
        "style":req.style,
        "tip":"Be specific & personal"
    }

class BatchIn(BaseModel):
    text:str

@app.post("/rewrite-batch")
async def rewrite_batch(req:BatchIn):
    sents=split_sentences(req.text)
    new=[x+" (humanized)" for x in sents]
    return {"rewritten":stitch(new)}

class PdfIn(BaseModel):
    text:str
    filename:str="report.pdf"

@app.post("/report.pdf")
async def report(req:PdfIn):
    buf=io.BytesIO()
    c=canvas.Canvas(buf,pagesize=letter)
    c.drawString(40,760,"AI Report - Downloaded")
    c.save();buf.seek(0)
    return StreamingResponse(buf,media_type="application/pdf",
        headers={"Content-Disposition":f'attachment; filename="{req.filename}"'}
    )
