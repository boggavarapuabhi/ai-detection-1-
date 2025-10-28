import re
import io
import asyncio
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
from rapidfuzz import fuzz

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

app = FastAPI(title="AI Text & Plagiarism Detector", version="1.10.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- utils ----------
_SENTENCE_RE = re.compile(r'(?us)([^.!?]+[.!?])')
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.findall(text) or [text]
    return [p.strip() for p in parts if p.strip()]

def stitch_text(sentences: List[str]) -> str:
    # Sentences already include punctuation; just join with spaces
    return " ".join(s.strip() for s in sentences if s.strip())

def normalize_text(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()

def ai_score_simple(text: str) -> int:
    sents = split_sentences(text)
    long_count = sum(len(s.split()) >= 18 for s in sents)
    return max(0, min(100, 15 + long_count * 12))

# ---------- plagiarism helpers ----------
async def fetch_url_text(url: str, client: httpx.AsyncClient) -> str:
    try:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        return normalize_text(r.text)
    except Exception:
        return ""

async def build_corpus(source_texts: List[str], source_urls: List[str]) -> List[Tuple[str, str]]:
    corpus: List[Tuple[str, str]] = []
    for i, t in enumerate(source_texts or []):
        if t and t.strip():
            corpus.append((f"text:{i}", normalize_text(t)))
    if source_urls:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            results = await asyncio.gather(*[fetch_url_text(u, client) for u in source_urls])
        for u, body in zip(source_urls, results):
            if body:
                corpus.append((u, body))
    return corpus

def best_match_for_sentence(sent: str, corp_text: str) -> Tuple[int, str]:
    best_score, best_snip = 0, ""
    for cs in split_sentences(corp_text):
        score = max(fuzz.token_set_ratio(sent, cs), fuzz.partial_ratio(sent, cs))
        if score > best_score:
            best_score, best_snip = score, cs
    return best_score, best_snip

def reason_for_ai_like(sentence: str) -> str:
    words = sentence.split()
    long = len(words) >= 22
    low_var = len(set(w.lower().strip(",.?!;:") for w in words)) / max(1, len(words)) < 0.65
    passive = any(p in sentence.lower() for p in ["is being", "was", "were", "has been", "have been"])
    bits = []
    if long: bits.append("very long")
    if low_var: bits.append("low lexical variety")
    if passive: bits.append("passive tone")
    return ", ".join(bits) if bits else "generic phrasing"

def suggestion_for(label: str, sentence: str) -> str:
    if label == "ai":
        return "Shorten, add concrete detail, first-person perspective, and vary verbs."
    if label == "plag":
        return "Paraphrase in your own words and cite the source."
    return "Looks good. Keep the specific context and voice."

# ---------- models ----------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Plain text to analyze")
    source_texts: Optional[List[str]] = Field(default=None, description="Optional reference texts")
    source_urls: Optional[List[str]]  = Field(default=None, description="Optional URLs to compare")
    plag_threshold: int = Field(default=85, ge=50, le=100)

class SentenceOut(BaseModel):
    index: int
    text: str
    label: str                 # ai | plag | human
    confidence: str            # Low | Medium | High
    ai_like: Optional[bool] = None
    plag_score: Optional[int] = None
    source: Optional[str] = None
    source_snippet: Optional[str] = None
    reason: Optional[str] = None
    suggestion: Optional[str] = None

class AnalyzeJSON(BaseModel):
    summary: Dict[str, int]
    counts: Dict[str, int]
    sentences: List[SentenceOut]
    markdown_report: str

# ---------- core analyzer ----------
async def analyze_core(req: AnalyzeRequest) -> AnalyzeJSON:
    text = req.text.strip()
    sentences = split_sentences(text)
    ai_overall = ai_score_simple(text)
    plag_threshold = req.plag_threshold

    corpus = await build_corpus(req.source_texts or [], req.source_urls or [])

    out: List[SentenceOut] = []
    seen = set()
    for i, s in enumerate(sentences):
        is_dupe = s in seen
        seen.add(s)

        best_plag_score, best_source, best_snip = 0, None, None
        for src_id, body in corpus:
            sc, sn = best_match_for_sentence(s, body)
            if sc > best_plag_score:
                best_plag_score, best_source, best_snip = sc, src_id, sn

        ai_like = (len(s.split()) >= 18) or (ai_overall >= 70)
        label = "human"
        if is_dupe:
            best_plag_score = max(best_plag_score, 70 if len(s.split()) > 8 else 0)
        if best_plag_score >= plag_threshold:
            label = "plag"
        elif ai_like:
            label = "ai"

        # confidence
        conf = "Low"
        if label == "plag":
            conf = "High" if best_plag_score >= 92 else "Medium"
        elif label == "ai":
            conf = "High" if ai_overall >= 80 and len(s.split()) >= 22 else "Medium"

        out.append(SentenceOut(
            index=i,
            text=s,
            label=label,
            confidence=conf,
            ai_like=ai_like,
            plag_score=best_plag_score if best_plag_score else None,
            source=best_source,
            source_snippet=best_snip,
            reason=reason_for_ai_like(s) if label == "ai" else ("high similarity" if label == "plag" else None),
            suggestion=suggestion_for(label, s)
        ))

    plag_sent = sum(1 for o in out if o.label == "plag")
    ai_sent   = sum(1 for o in out if o.label == "ai")
    total     = max(1, len(out))

    summary = {
        "ai_score": ai_overall,
        "plagiarism_score": min(100, int(100 * plag_sent / total)),
        "human_score": max(0, 100 - max(ai_overall, int(100 * plag_sent / total))),
    }
    counts = {"total": total, "ai": ai_sent, "plag": plag_sent, "human": total - ai_sent - plag_sent}

    # Markdown report for GPT display
    def md_section(title: str) -> str:
        return f"\n### {title}\n"

    md = []
    md.append("## AI Text Detection Report")
    md.append(md_section("Overall Results"))
    md.append(f"- **AI-likeness:** **{summary['ai_score']}%**")
    md.append(f"- **Plagiarism risk:** **{summary['plagiarism_score']}%**")
    md.append(f"- **Human-likeness:** **{summary['human_score']}%**")
    if counts["ai"] > 0:
        md.append(md_section("AI-flagged sentences"))
        for o in out:
            if o.label == "ai":
                md.append(f"**{o.index+1}.** {o.text}\n  - Reason: {o.reason}\n  - Confidence: {o.confidence}\n  - Fix: {o.suggestion}")
    else:
        md.append(md_section("AI-flagged sentences"))
        md.append("None detected ✅")
    if counts["plag"] > 0:
        md.append(md_section("Plagiarism-risk sentences"))
        for o in out:
            if o.label == "plag":
                src = f" ({o.source})" if o.source else ""
                note = f"Matched snippet: “{o.source_snippet}”" if o.source_snippet else ""
                md.append(f"**{o.index+1}.** {o.text}\n  - Similarity score: {o.plag_score}%{src}\n  - Confidence: {o.confidence}\n  - {note}\n  - Fix: {o.suggestion}")
    else:
        md.append(md_section("Plagiarism-risk sentences"))
        md.append("None detected ✅")
    md.append(md_section("Notes"))
    md.append("AI and plagiarism results are probabilistic indicators. Use human judgment and cite sources where appropriate.")
    markdown_report = "\n".join(md)

    return AnalyzeJSON(summary=summary, counts=counts, sentences=out, markdown_report=markdown_report)

# ---------- endpoints ----------
@app.post("/analyze", response_model=AnalyzeJSON)
async def analyze(req: AnalyzeRequest):
    result = await analyze_core(req)
    return JSONResponse(content=result.model_dump())

# Simple HTML highlight preview for YOUR website (not for ChatGPT)
def _build_html(sentences: List[SentenceOut]) -> str:
    from html import escape as _esc
    colors = {"ai": "rgba(255,87,87,.25)", "plag": "rgba(255,193,7,.35)", "human": "rgba(76,175,80,.18)"}
    spans = []
    for s in sentences:
        title = f"{(s.source or '')} score={s.plag_score}" if s.label == "plag" else s.reason or ""
        spans.append(
            f'<span title="{_esc(title)}" style="background:{colors[s.label]};padding:2px 3px;border-radius:4px" '
            f'data-index="{s.index}">{_esc(s.text)}</span>'
        )
    return "<div style='font-family:ui-sans-serif,system-ui;line-height:1.6'>" + " ".join(spans) + "</div>"

@app.post("/preview", response_class=HTMLResponse)
async def preview(req: AnalyzeRequest):
    result = await analyze_core(req)
    return HTMLResponse(content=_build_html(result.sentences))

# ---------- REWRITE (single) ----------
class RewriteRequest(BaseModel):
    sentence: str
    style: str = Field("humanize", description="humanize | concise | formal | simple")

class RewriteOut(BaseModel):
    original: str
    style: str
    rewritten: str
    tips: str

def _rewrite_rule_based(sentence: str, style: str) -> str:
    s = sentence.strip()
    s = re.sub(r"\s+", " ", s)

    if style == "concise":
        s = re.sub(r"\b(in order to|due to the fact that|with respect to|it is important to note that)\b", "to", s, flags=re.I)
        s = re.sub(r"\b(that|which)\b", "", s, count=1)
        words = s.split()
        if len(words) > 28:
            s = " ".join(words[:24]) + "."
    elif style == "formal":
        s = re.sub(r"\b(can't|won't|don't|isn't|aren't|doesn't)\b", lambda m: {
            "can't":"cannot","won't":"will not","don't":"do not","isn't":"is not",
            "aren't":"are not","doesn't":"does not"
        }[m.group(0)], s, flags=re.I)
        s = s.replace(" kids ", " children ")
    elif style == "simple":
        s = re.sub(r",\s*", ". ", s)
        s = re.sub(r"\butilize\b", "use", s, flags=re.I)
    else:  # humanize
        s = re.sub(r"\b(is|was|were|been|being|are) ([a-z]+)ed\b", r"\2", s)
        if s.endswith("."):
            s = s[:-1] + " with a concrete example."
        else:
            s += " with a concrete example."
    return s.strip()

@app.post("/rewrite", response_model=RewriteOut)
async def rewrite(req: RewriteRequest):
    style = req.style.lower()
    if style not in {"humanize", "concise", "formal", "simple"}:
        style = "humanize"
    new_sentence = _rewrite_rule_based(req.sentence, style)
    tips = {
        "humanize": "Add personal insight, specific nouns, and varied verbs.",
        "concise": "Remove fillers, tighten clauses, prefer strong verbs over adverbs.",
        "formal": "Prefer full forms, objective tone, and precise vocabulary.",
        "simple": "Use short sentences, common words, and avoid jargon."
    }[style]
    return JSONResponse(content=RewriteOut(
        original=req.sentence, style=style, rewritten=new_sentence, tips=tips
    ).model_dump())

# ---------- REWRITE BATCH (document) ----------
class RewriteBatchRequest(BaseModel):
    text: str
    style: str = Field("humanize", description="default style if style_map not provided")
    style_map: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional per-label style, keys in {ai,plag,human}"
    )
    target_labels: Optional[List[str]] = Field(
        default=["ai", "plag"],
        description="Which labels to rewrite; default rewrites flagged only"
    )
    plag_threshold: int = Field(default=85, ge=50, le=100)
    source_texts: Optional[List[str]] = None
    source_urls: Optional[List[str]] = None

class RewriteBatchOut(BaseModel):
    replaced_count: int
    originals: List[Dict]
    rewrites: List[Dict]
    rewritten_text_flagged_only: str
    rewritten_text_all: str

@app.post("/rewrite-batch", response_model=RewriteBatchOut)
async def rewrite_batch(req: RewriteBatchRequest):
    # Run analysis to know which sentences to touch
    analysis = await analyze_core(AnalyzeRequest(
        text=req.text,
        source_texts=req.source_texts,
        source_urls=req.source_urls,
        plag_threshold=req.plag_threshold
    ))
    sents = [s.text for s in analysis.sentences]
    labels = [s.label for s in analysis.sentences]

    # Determine style per label
    style_map = {**{"ai": req.style, "plag": req.style, "human": req.style}, **(req.style_map or {})}

    originals, rewrites = [], []
    rewritten_flagged = sents[:]
    rewritten_all = sents[:]
    replaced_count = 0

    for i, (sent, lab) in enumerate(zip(sents, labels)):
        pick_style = style_map.get(lab, req.style)
        new = _rewrite_rule_based(sent, pick_style)

        # Record items
        obj_in = {"index": i, "label": lab, "style": pick_style, "text": sent}
        obj_out = {"index": i, "label": lab, "style": pick_style, "rewritten": new}
        originals.append(obj_in)
        rewrites.append(obj_out)

        # Replace only if target label
        if lab in (req.target_labels or []):
            rewritten_flagged[i] = new
            replaced_count += 1
        # Replace all for alt draft
        rewritten_all[i] = new

    return JSONResponse(content=RewriteBatchOut(
        replaced_count=replaced_count,
        originals=originals,
        rewrites=rewrites,
        rewritten_text_flagged_only=stitch_text(rewritten_flagged),
        rewritten_text_all=stitch_text(rewritten_all)
    ).model_dump())

# ---------- PDF REPORT endpoint ----------
class PdfRequest(BaseModel):
    text: str
    source_texts: Optional[List[str]] = None
    source_urls: Optional[List[str]] = None
    plag_threshold: int = Field(default=85, ge=50, le=100)
    filename: Optional[str] = "detector_report.pdf"

def _draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, leading: float = 14):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split(" ")
    line = ""
    while words:
        w = words.pop(0)
        probe = f"{line} {w}".strip()
        if stringWidth(probe, "Helvetica", 10) <= width:
            line = probe
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y

@app.post("/report.pdf")
async def report_pdf(req: PdfRequest):
    analysis = await analyze_core(AnalyzeRequest(
        text=req.text,
        source_texts=req.source_texts,
        source_urls=req.source_urls,
        plag_threshold=req.plag_threshold
    ))

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "AI Text Detection Report")
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, height - 1.2*inch, "This report is informational. Use human judgment and cite sources as needed.")

    y = height - 1.5*inch
    left = 1*inch
    right_limit = width - 1*inch
    text_width = right_limit - left

    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Overall Results")
    y -= 16
    c.setFont("Helvetica", 10)
    y = _draw_wrapped(c, f"AI-likeness: {analysis.summary['ai_score']}%", left, y, text_width)
    y = _draw_wrapped(c, f"Plagiarism risk: {analysis.summary['plagiarism_score']}%", left, y, text_width)
    y = _draw_wrapped(c, f"Human-likeness: {analysis.summary['human_score']}%", left, y, text_width)
    y -= 6

    # AI-flagged
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "AI-flagged sentences")
    y -= 16
    c.setFont("Helvetica", 10)
    ai_items = [o for o in analysis.sentences if o.label == "ai"]
    if not ai_items:
        y = _draw_wrapped(c, "None detected", left, y, text_width)
    else:
        for o in ai_items:
            y = _draw_wrapped(c, f"{o.index+1}. {o.text}", left, y, text_width)
            y = _draw_wrapped(c, f"   Reason: {o.reason} | Confidence: {o.confidence}", left, y, text_width)
            y = _draw_wrapped(c, f"   Fix: {o.suggestion}", left, y, text_width)
            y -= 6
            if y < 1*inch:
                c.showPage(); y = height - 1*inch

    # Plagiarism
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Plagiarism-risk sentences")
    y -= 16
    c.setFont("Helvetica", 10)
    plag_items = [o for o in analysis.sentences if o.label == "plag"]
    if not plag_items:
        y = _draw_wrapped(c, "None detected", left, y, text_width)
    else:
        for o in plag_items:
            src = f" ({o.source})" if o.source else ""
            y = _draw_wrapped(c, f"{o.index+1}. {o.text}", left, y, text_width)
            y = _draw_wrapped(c, f"   Similarity: {o.plag_score}%{src} | Confidence: {o.confidence}", left, y, text_width)
            if o.source_snippet:
                y = _draw_wrapped(c, f"   Match: {o.source_snippet}", left, y, text_width)
            y = _draw_wrapped(c, f"   Fix: {o.suggestion}", left, y, text_width)
            y -= 6
            if y < 1*inch:
                c.showPage(); y = height - 1*inch

    c.showPage()
    c.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{req.filename or "detector_report.pdf"}"'}
    )
