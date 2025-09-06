from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import os, time, re, logging, concurrent.futures, traceback

import gradio as gr
from gradio.routes import mount_gradio_app


from app.utils.llm import ask_question, ask_arxiv_question

# Config via ENV

FILE_MAX_MB: int = int(os.getenv("FILE_MAX_MB", "5"))

LATENCY_TIMEOUT_MS: int = int(os.getenv("LATENCY_TIMEOUT_MS", "900000"))  
LATENCY_DEFAULT_MS: int = int(os.getenv("LATENCY_DEFAULT_MS", "60000"))   
LATENCY_MAX_MS: int = int(os.getenv("LATENCY_MAX_MS", "600000"))          

REQUEST_TIMEOUT_MS: int = int(os.getenv("REQUEST_TIMEOUT_MS", "60000"))   
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

CPU_FALLBACK_MODEL: str = os.getenv("CPU_FALLBACK_MODEL", "microsoft/Phi-3-mini-4k-instruct")

DEFAULT_MAX_INPUT_TOKENS = int(os.getenv("DEFAULT_MAX_INPUT_TOKENS", "8000"))
MODEL_MAX_INPUT_TOKENS: Dict[str, int] = {
    "microsoft/Phi-3-mini-4k-instruct": 4000,
    "tiiuae/falcon-7b-instruct": 4096,
    "Qwen/Qwen2.5-7B-Instruct": 8000,
    "mistralai/Mistral-7B-Instruct-v0.2": 8000,
    "mistralai/Mistral-7B-Instruct-v0.3": 8000,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm_api")


try:
    import torch
    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    torch = None
    _HAS_CUDA = False
log.info("CUDA available: %s", _HAS_CUDA)


# models

class Mode(str, Enum):
    short = "short"
    detailed = "detailed"

class ModelName(str, Enum):
    mistral_v02 = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral_v03 = "mistralai/Mistral-7B-Instruct-v0.3"
    phi3_mini  = "microsoft/Phi-3-mini-4k-instruct"
    qwen_7b    = "Qwen/Qwen2.5-7B-Instruct"
    falcon_7b  = "tiiuae/falcon-7b-instruct"

MODEL_IDS: List[str] = [m.value for m in ModelName]

MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "nickname": "Mistral v0.2 (7B)",
        "best_for": ["General QA (short–medium)", "Brief summaries", "Balanced quality/speed"],
        "input_recommendation": "Short to medium texts; focused questions.",
        "tone": "Concise and fairly accurate.",
        "notes": "Stable daily-driver.",
        "link": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "nickname": "Mistral v0.3 (7B)",
        "best_for": ["More complex instructions", "Summary + short rationale"],
        "input_recommendation": "Medium input; a few steps.",
        "tone": "Follows instructions well.",
        "notes": "Improved phrasing vs. v0.2.",
        "link": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "nickname": "Phi-3 Mini (4k)",
        "best_for": ["Very short questions", "Quick chat", "Low-resource setups"],
        "input_recommendation": "Short snippets.",
        "tone": "Fast and economical.",
        "notes": "Short context window.",
        "link": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "nickname": "Qwen2.5 (7B)",
        "best_for": ["Bullet points", "Light tech/code", "Structured outputs"],
        "input_recommendation": "Medium input; structured tasks.",
        "tone": "Organized.",
        "notes": "Strong for a 7B model.",
        "link": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    },
    "tiiuae/falcon-7b-instruct": {
        "nickname": "Falcon 7B Instruct",
        "best_for": ["Basic use", "General short questions", "Fallback/variety"],
        "input_recommendation": "Short–medium input.",
        "tone": "Simple and to the point.",
        "notes": "Older model; variable quality.",
        "link": "https://huggingface.co/tiiuae/falcon-7b-instruct",
    },
}


class ArxivMetadata(BaseModel):
    title: str
    abstract: Optional[str] = ""
    authors: Optional[str] = ""

class LLMQueryInput(BaseModel):
    paper: Optional[str] = None
    metadata: Optional[ArxivMetadata] = None
    query: str

class AnswerResponse(BaseModel):
    answer: str
    meta: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None


# Token + RAG helpers

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0

def model_max_ctx(model_id: str) -> int:
    return MODEL_MAX_INPUT_TOKENS.get(model_id, DEFAULT_MAX_INPUT_TOKENS)

def output_reserve(mode: Mode) -> int:
    return 800 if mode == Mode.detailed else 400

def instruction_reserve() -> int:
    return 200

def split_into_chunks(text: str, chunk_chars: int = 1000, overlap: int = 150) -> List[str]:
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = ((buf[-overlap:] + "\n" + p).strip() if buf else p)
            while len(buf) > chunk_chars:
                chunks.append(buf[:chunk_chars])
                buf = buf[chunk_chars - overlap:]
    if buf:
        chunks.append(buf)
    return chunks

def simple_scores(question: str, chunks: List[str]) -> List[Tuple[int, float]]:
    q = re.findall(r"\w+", (question or "").lower())
    q_set = set(q)
    scores: List[Tuple[int, float]] = []
    for i, c in enumerate(chunks):
        t = c.lower()
        tf = sum(t.count(term) for term in q_set if len(term) > 2)
        overlap = len(q_set.intersection(set(re.findall(r"\w+", t))))
        scores.append((i, tf + 0.3 * overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def select_top_chunks_fit(chunks: List[str], question: str, budget_tokens: int, base_k: int = 5) -> List[Tuple[int, str]]:
    ranked = simple_scores(question, chunks)
    for k in range(base_k, 0, -1):
        sel, used = [], 0
        for i, _ in ranked[:k]:
            c = chunks[i]
            t = estimate_tokens(c) + 10
            if used + t <= budget_tokens:
                sel.append((i, c))
                used += t
        if sel:
            return sel
    if ranked:
        i, _ = ranked[0]
        c = chunks[i]
        max_chars = max(200, budget_tokens * 4 - 100)
        return [(i, c[:max_chars])]
    return []

def build_snippets_text(chunks_with_idx: List[Tuple[int, str]]) -> str:
    return "\n\n".join(f"[Snippet {i}]\n{c}" for i, c in chunks_with_idx)

def build_augmented_question(user_q: str) -> str:
    return ("Use ONLY the snippets provided to answer. "
            "Cite snippet numbers like [Snippet 3]. Be concise.\n\n"
            f"Question: {user_q}")

def prepare_context_from_paper(paper: str, question: str, model_id: str, mode: Mode) -> Dict[str, Any]:
    max_ctx = model_max_ctx(model_id)
    prompt_budget = max_ctx - output_reserve(mode) - instruction_reserve()
    if prompt_budget <= 300:
        raise HTTPException(status_code=413, detail=f"Model context too small (max {max_ctx}).")
    chunks = split_into_chunks(paper, 1000, 150)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty paper content.")

    selected = select_top_chunks_fit(chunks, question, int(0.85 * prompt_budget), 5)
    snippets = build_snippets_text(selected)
    aug_q = build_augmented_question(question)

    t_ctx = estimate_tokens(snippets)
    t_q = estimate_tokens(aug_q)
    t_total = t_ctx + t_q
    diag = {
        "max_ctx": max_ctx, "prompt_budget": prompt_budget,
        "tokens_context": t_ctx, "tokens_question": t_q, "tokens_total": t_total,
        "tokens_reserved_output": output_reserve(mode),
        "selected_snippets": [i for i, _ in selected], "selected_count": len(selected),
        "cannot_fit": t_total > prompt_budget
    }
    if t_total > prompt_budget:
        max_chars = max(1000, (prompt_budget - t_q) * 4 - 100)
        snippets = snippets[:max_chars]
        t_ctx = estimate_tokens(snippets)
        t_total = t_ctx + t_q
        diag["tokens_context"] = t_ctx
        diag["tokens_total"] = t_total
        if t_total > prompt_budget:
            raise HTTPException(status_code=413, detail=f"Input too long: ~{t_total} > {prompt_budget} tokens.")
    return {"snippets_context": snippets, "augmented_question": aug_q, "diagnostics": diag}


# threading + timeouts

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, len(MODEL_IDS)))

def _run_with_timeout(fn, timeout_ms: int, *args, **kwargs):
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_ms / 1000.0)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"operation timed out after {timeout_ms} ms")

def clamp_timeout(ms: int) -> int:
    """
     LATENCY_TIMEOUT_MS (hard clamp).
    """
    return 1000 if ms < 1000 else LATENCY_TIMEOUT_MS if ms > LATENCY_TIMEOUT_MS else ms

# Safe call wrappers: auto-fallback on CPU/meta errors

_META_ERR_SNIPPETS = (
    "is on the meta device, we need a `value` to put in on",
    "meta device",
    "Tensor.item() cannot be called on meta tensors",  # ← added
)
_CPU_7B_HINTS = (
    "out of memory",
    "CUDA out of memory",
    "no cuda",
    "CUDA is not available",
    "device_map",
)

def _looks_like_meta_cpu_error(exc: Exception) -> bool:
    msg = f"{exc}"
    if any(s in msg for s in _META_ERR_SNIPPETS):
        return True
    if any(s in msg.lower() for s in _CPU_7B_HINTS):
        return True
        
    tb = "".join(traceback.format_exception_only(type(exc), exc))
    return "accelerate/hooks.py" in tb or "set_module_tensor_to_device" in tb

def _call_ask_question_once(context: str, question: str, model_id: Optional[str], mode: Optional[str]):

    try:
        return ask_question(context=context, question=question, model_key=model_id, mode=mode)
    except TypeError:
        try:
            return ask_question(context=context, question=question, model_name=model_id, mode=mode)
        except TypeError:
            try:
                return ask_question(context=context, question=question, model=model_id, mode=mode)
            except TypeError:
                return ask_question(context=context, question=question)

def _call_ask_arxiv_once(metadata: Dict[str, Any], question: str, model_id: Optional[str], mode: Optional[str]):
    try:
        return ask_arxiv_question(metadata=metadata, question=question, model_key=model_id, mode=mode)
    except TypeError:
        try:
            return ask_arxiv_question(metadata=metadata, question=question, model_name=model_id, mode=mode)
        except TypeError:
            try:
                return ask_arxiv_question(metadata=metadata, question=question, model=model_id, mode=mode)
            except TypeError:
                return ask_arxiv_question(metadata=metadata, question=question)

def _call_ask_question(context: str, question: str, model_id: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:

    fallback_used = False
    fallback_reason = ""
    chosen_model = model_id or CPU_FALLBACK_MODEL

    try:
        ans = _call_ask_question_once(context, question, chosen_model, mode)
        return {"answer": ans, "fallback_used": fallback_used, "fallback_reason": fallback_reason, "chosen_model": chosen_model}
    except Exception as e:
        if _looks_like_meta_cpu_error(e):
            # נסיון פולבאק
            fallback_used = True
            fallback_reason = f"auto-fallback due to CPU/meta error: {str(e)[:140]}"
            try:
                ans = _call_ask_question_once(context, question, CPU_FALLBACK_MODEL, mode)
                return {"answer": ans, "fallback_used": fallback_used, "fallback_reason": fallback_reason, "chosen_model": CPU_FALLBACK_MODEL}
            except Exception as e2:
                raise e2
        else:
            raise e

def _call_ask_arxiv(metadata: Dict[str, Any], question: str, model_id: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:
    fallback_used = False
    fallback_reason = ""
    chosen_model = model_id or CPU_FALLBACK_MODEL
    try:
        ans = _call_ask_arxiv_once(metadata, question, chosen_model, mode)
        return {"answer": ans, "fallback_used": fallback_used, "fallback_reason": fallback_reason, "chosen_model": chosen_model}
    except Exception as e:
        if _looks_like_meta_cpu_error(e):
            fallback_used = True
            fallback_reason = f"auto-fallback due to CPU/meta error: {str(e)[:140]}"
            try:
                ans = _call_ask_arxiv_once(metadata, question, CPU_FALLBACK_MODEL, mode)
                return {"answer": ans, "fallback_used": fallback_used, "fallback_reason": fallback_reason, "chosen_model": CPU_FALLBACK_MODEL}
            except Exception as e2:
                raise e2
        else:
            raise e

def sanitize_model_id(val: str) -> str:

    return val if val in MODEL_IDS else ModelName.mistral_v02.value

# Latency functions 

def _ping_once(model_id: str, timeout_ms: int) -> Optional[float]:
    try:
        t0 = time.perf_counter()

        _ = _run_with_timeout(_call_ask_question, clamp_timeout(timeout_ms), "", "ping", model_id, "short")
        return time.perf_counter() - t0
    except Exception:
        return None

def measure_latency(model_id: str, max_wait_ms: int = LATENCY_DEFAULT_MS) -> Optional[float]:
    mid = sanitize_model_id(model_id)
    return _ping_once(mid, max_wait_ms)

def measure_latencies_all(model_ids: List[str],
                          soft_timeout_ms: int = LATENCY_DEFAULT_MS,
                          ensure_one_model: Optional[str] = None,
                          ensure_timeout_ms: int = LATENCY_MAX_MS) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {}
    if not model_ids:
        return results
    max_workers = min(8, max(4, len(model_ids)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {pool.submit(_ping_once, mid, soft_timeout_ms): mid for mid in model_ids}
        for fut in concurrent.futures.as_completed(fut_map):
            mid = fut_map[fut]
            try:
                results[mid] = fut.result()
            except Exception:
                results[mid] = None
    if all(v is None for v in results.values()):
        target = ensure_one_model or ModelName.mistral_v02.value
        results[target] = _ping_once(target, ensure_timeout_ms)
    return results


# Help & latency text builders

def build_help_markdown_static() -> str:
    lines = []
    lines.append("#  Model Help & Comparison\n")
    lines.append(f"Hardware detected: **{'GPU' if _HAS_CUDA else 'CPU-only'}**. All models are selectable.\n")
    lines.append(f"If a heavy model fails to load on CPU, the app auto-falls back to `{CPU_FALLBACK_MODEL}` and notes it in the run report.\n")
    lines.append("| Model | Nickname | Best for | Recommended Input | Tone | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for mid in MODEL_IDS:
        info = MODEL_INFO.get(mid, {})
        name_cell = f"[`{mid}`]({info.get('link', '#')})"
        best_for = " • ".join(info.get("best_for", []))
        lines.append(f"| {name_cell} | {info.get('nickname','')} | {best_for} | "
                     f"{info.get('input_recommendation','')} | {info.get('tone','')} | {info.get('notes','')} |")
    return "\n".join(lines)

def build_help_with_latency(model_id: str, wait_ms: int) -> str:
    base = build_help_markdown_static()
    mid = sanitize_model_id(model_id)
    lat = measure_latency(mid, wait_ms)
    if lat is None:
        note = f"**Live latency for `{mid}`:** timeout / unavailable (tried ~{int(wait_ms/1000)}s)"
    else:
        note = f"**Live latency for `{mid}`:** {lat:.2f} s (timeout set to {int(wait_ms/1000)}s)"
    return base + "\n\n" + note

def build_help_with_all_latencies(wait_ms: int) -> str:
    base = build_help_markdown_static()
    res = measure_latencies_all(MODEL_IDS, soft_timeout_ms=wait_ms)
    lines = []
    lines.append("\n\n**Live latency (s):**\n")
    lines.append("| Model | Latency |")
    lines.append("|---|---|")
    for mid in MODEL_IDS:
        v = res.get(mid)
        cell = f"{v:.2f}" if isinstance(v, float) and v is not None else "timeout"
        lines.append(f"| `{mid}` | {cell} |")
    return base + "\n" + "\n".join(lines)


# FastAPI app

app = FastAPI(title="LLM Query API", version="1.8", docs_url="/docs", redoc_url="/redoc")

origins = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=str(exc.detail)).model_dump())

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content=ErrorResponse(error="Internal server error", details=str(exc)[:200]).model_dump())

@app.get("/", response_model=dict)
def root():
    return {"message": "API is running!"}

@app.get("/health", response_model=dict)
def health():
    return {"status": "ok"}

@app.get("/info", response_model=dict)
def info():
    return {
        "cuda": bool(_HAS_CUDA),
        "available_models": MODEL_IDS,
        "cpu_fallback_model": CPU_FALLBACK_MODEL,
        "defaults": {
            "request_timeout_ms": REQUEST_TIMEOUT_MS,
            "latency_default_ms": LATENCY_DEFAULT_MS,
            "latency_max_ms": LATENCY_MAX_MS,
        }
    }

# Endpoints

def _format_run_report(meta: Dict[str, Any]) -> str:
    t = meta.get("timing_ms", {})
    tok = meta.get("tokens", {})
    lines = []
    lines.append("### Run report")
    lines.append(f"- Path: **{meta.get('path','')}**")
    lines.append(f"- Requested model: **{meta.get('requested_model','')}**")
    lines.append(f"- Used model: **{meta.get('model_id','')}**")
    if meta.get("fallback_used"):
        lines.append(f"- Fallback used: **Yes**")
        if meta.get("fallback_reason"):
            lines.append(f"  - Reason: {meta.get('fallback_reason')}")
    else:
        lines.append(f"- Fallback used: **No**")
    lines.append(f"- Mode: **{meta.get('mode','')}**")
    lines.append(f"- Timeout used: **{meta.get('request_timeout_ms_used','')} ms** (~{meta.get('timeout_seconds_used','?')} s)")
    if t:
        lines.append(f"- Timing: prep **{t.get('prep',0)} ms**, LLM **{t.get('llm',0)} ms**, total **{t.get('total',0)} ms**")
    if tok:
        if "prompt_budget" in tok:
            lines.append(f"- Tokens: ctx ≈ **{tok.get('tokens_context','?')}**, question ≈ **{tok.get('tokens_question','?')}**, total ≈ **{tok.get('tokens_total','?')}** / budget **{tok.get('prompt_budget','?')}** (max ctx **{tok.get('max_ctx','?')}**, reserved output **{tok.get('tokens_reserved_output','?')}**)")
        elif "approx_input_tokens" in tok:
            lines.append(f"- Tokens (metadata): input ≈ **{tok.get('approx_input_tokens','?')}**, budget **{tok.get('prompt_budget','?')}**, max ctx **{tok.get('max_ctx','?')}**")
        if "selected_snippets" in tok:
            lines.append(f"- Selected snippets: {tok.get('selected_snippets',[])} (count {tok.get('selected_count',0)})")
        if tok.get("cannot_fit"):
            lines.append("⚠️ Some content was trimmed to fit the prompt budget.")
    return "\n".join(lines)

from fastapi import Body

@app.post("/uploadfile", include_in_schema=True, response_model=AnswerResponse)
async def upload_file(file: UploadFile = File(..., description="Upload .txt file"), query: str = Query(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    contents = await file.read()
    if len(contents) > FILE_MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (> {FILE_MAX_MB} MB)")
    text = contents.decode("utf-8", errors="ignore")

    requested_model = ModelName.mistral_v02.value  # UI-less endpoint – מבקשים ברירת מחדל "גבוהה"
    model_id = sanitize_model_id(requested_model)
    mode_val = Mode.short
    timeout = clamp_timeout(REQUEST_TIMEOUT_MS)

    t0 = time.perf_counter()
    prep = prepare_context_from_paper(text, query, model_id=model_id, mode=mode_val)
    t1 = time.perf_counter()
    call_res = _run_with_timeout(_call_ask_question, timeout, prep["snippets_context"], prep["augmented_question"], model_id, mode_val.value)
    t2 = time.perf_counter()

    used_model = call_res.get("chosen_model", model_id)
    meta = {
        "path": "uploadfile",
        "requested_model": requested_model,
        "model_id": used_model,
        "mode": mode_val.value,
        "fallback_used": call_res.get("fallback_used", False),
        "fallback_reason": call_res.get("fallback_reason", ""),
        "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
        "timing_ms": {"prep": int((t1 - t0) * 1000), "llm": int((t2 - t1) * 1000), "total": int((t2 - t0) * 1000)},
        "tokens": prep["diagnostics"],
    }
    return AnswerResponse(answer=str(call_res["answer"]), meta=meta)

@app.post("/query", include_in_schema=True, response_model=AnswerResponse)
async def query_llm(
    input: LLMQueryInput = Body(...),
    mode: Mode = Query(Mode.short, description="Answer length"),
    model_name: ModelName = Query(ModelName.mistral_v02, description="Choose the LLM"),
    request_timeout_ms: int = Query(REQUEST_TIMEOUT_MS, ge=1000, le=900000, description="Per-request timeout in ms")
):
    requested_model = model_name.value
    model_id = sanitize_model_id(requested_model)
    mode_val = mode
    timeout = clamp_timeout(request_timeout_ms)

    if input.metadata:
        t0 = time.perf_counter()
        md = input.metadata.model_dump()
        call_res = _run_with_timeout(_call_ask_arxiv, timeout, md, input.query, model_id, mode_val.value)
        t1 = time.perf_counter()
        used_model = call_res.get("chosen_model", model_id)
        budget = model_max_ctx(used_model) - output_reserve(mode_val) - instruction_reserve()
        meta = {
            "path": "metadata", "requested_model": requested_model, "model_id": used_model, "mode": mode_val.value,
            "fallback_used": call_res.get("fallback_used", False),
            "fallback_reason": call_res.get("fallback_reason", ""),
            "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
            "timing_ms": {"prep": 0, "llm": int((t1 - t0) * 1000), "total": int((t1 - t0) * 1000)},
            "tokens": {
                "approx_input_tokens": estimate_tokens(" ".join([md.get("title", ""), md.get("abstract", ""), md.get("authors", "")])),
                "max_ctx": model_max_ctx(used_model), "prompt_budget": budget, "tokens_reserved_output": output_reserve(mode_val),
            },
        }
        return AnswerResponse(answer=str(call_res["answer"]), meta=meta)

    if input.paper:
        t0 = time.perf_counter()
        prep = prepare_context_from_paper(input.paper, input.query, model_id=model_id, mode=mode_val)
        t1 = time.perf_counter()
        call_res = _run_with_timeout(_call_ask_question, timeout, prep["snippets_context"], prep["augmented_question"], model_id, mode_val.value)
        t2 = time.perf_counter()
        used_model = call_res.get("chosen_model", model_id)
        meta = {
            "path": "paper", "requested_model": requested_model, "model_id": used_model, "mode": mode_val.value,
            "fallback_used": call_res.get("fallback_used", False),
            "fallback_reason": call_res.get("fallback_reason", ""),
            "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
            "timing_ms": {"prep": int((t1 - t0) * 1000), "llm": int((t2 - t1) * 1000), "total": int((t2 - t0) * 1000)},
            "tokens": prep["diagnostics"],
        }
        return AnswerResponse(answer=str(call_res["answer"]), meta=meta)

    raise HTTPException(status_code=400, detail="Provide either 'paper' or 'metadata'")

@app.post("/query_form", response_model=AnswerResponse)
async def query_form(
    query: str = Form(..., description="Your question"),
    mode: Mode = Form(Mode.short, description="Answer length"),
    model_name: ModelName = Form(ModelName.mistral_v02, description="Choose the LLM"),
    paper: Optional[str] = Form(None, description="Full paper text (optional)"),
    metadata_title: Optional[str] = Form(None, description="arXiv Title"),
    metadata_abstract: Optional[str] = Form("", description="arXiv Abstract"),
    metadata_authors: Optional[str] = Form("", description="arXiv Authors"),
    file: Optional[UploadFile] = File(None, description="Upload .txt file"),
    request_timeout_ms: int = Form(REQUEST_TIMEOUT_MS),
):
    if file is not None:
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
        contents = await file.read()
        if len(contents) > FILE_MAX_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large (> {FILE_MAX_MB} MB)")
        paper = contents.decode("utf-8", errors="ignore")

    requested_model = model_name.value
    model_id = sanitize_model_id(requested_model)
    mode_val = mode
    timeout = clamp_timeout(request_timeout_ms)

    if metadata_title:
        t0 = time.perf_counter()
        md = {"title": metadata_title, "abstract": metadata_abstract or "", "authors": metadata_authors or ""}
        call_res = _run_with_timeout(_call_ask_arxiv, timeout, md, query, model_id, mode_val.value)
        t1 = time.perf_counter()
        used_model = call_res.get("chosen_model", model_id)
        budget = model_max_ctx(used_model) - output_reserve(mode_val) - instruction_reserve()
        meta = {
            "path": "metadata(form)", "requested_model": requested_model, "model_id": used_model, "mode": mode_val.value,
            "fallback_used": call_res.get("fallback_used", False),
            "fallback_reason": call_res.get("fallback_reason", ""),
            "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
            "timing_ms": {"prep": 0, "llm": int((t1 - t0) * 1000), "total": int((t1 - t0) * 1000)},
            "tokens": {
                "approx_input_tokens": estimate_tokens(" ".join([md["title"], md["abstract"], md["authors"]])),
                "max_ctx": model_max_ctx(used_model), "prompt_budget": budget, "tokens_reserved_output": output_reserve(mode_val),
            },
        }
        return AnswerResponse(answer=str(call_res["answer"]), meta=meta)

    if paper:
        t0 = time.perf_counter()
        prep = prepare_context_from_paper(paper, query, model_id=model_id, mode=mode_val)
        t1 = time.perf_counter()
        call_res = _run_with_timeout(_call_ask_question, timeout, prep["snippets_context"], prep["augmented_question"], model_id, mode_val.value)
        t2 = time.perf_counter()
        used_model = call_res.get("chosen_model", model_id)
        meta = {
            "path": "paper(form)", "requested_model": requested_model, "model_id": used_model, "mode": mode_val.value,
            "fallback_used": call_res.get("fallback_used", False),
            "fallback_reason": call_res.get("fallback_reason", ""),
            "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
            "timing_ms": {"prep": int((t1 - t0) * 1000), "llm": int((t2 - t1) * 1000), "total": int((t2 - t0) * 1000)},
            "tokens": prep["diagnostics"],
        }
        return AnswerResponse(answer=str(call_res["answer"]), meta=meta)

    raise HTTPException(status_code=400, detail="Provide either paper text/file or metadata (title/abstract/authors).")


# Gradio UI

def _token_info_for_paper(paper: str, question: str, model_id: str, mode: str) -> str:
    try:
        m = Mode(mode) if isinstance(mode, str) else mode
        prep = prepare_context_from_paper(paper or "", question or "", model_id=sanitize_model_id(model_id), mode=m)
        d = prep["diagnostics"]
        return (f"**Token budget**: max ctx ≈ {d['max_ctx']} | prompt budget ≈ {d['prompt_budget']} | "
                f"context ≈ {d['tokens_context']} | question ≈ {d['tokens_question']} | total ≈ {d['tokens_total']} "
                f"| selected {d['selected_count']} snippets {d['selected_snippets']}")
    except HTTPException as e:
        return f" {e.detail}"

def ui_submit(title, abstract, authors, paper, question, mode, model_id, request_timeout_ms):
    try:
        timeout = clamp_timeout(int(request_timeout_ms))
        model_id = sanitize_model_id(model_id)

        if title:
            metadata = {"title": title, "abstract": abstract or "", "authors": authors or ""}
            t0 = time.perf_counter()
            call_res = _run_with_timeout(_call_ask_arxiv, timeout, metadata, question, model_id, mode)
            t1 = time.perf_counter()
            used_model = call_res.get("chosen_model", model_id)
            meta = {
                "path": "metadata(ui)", "requested_model": model_id, "model_id": used_model, "mode": mode,
                "fallback_used": call_res.get("fallback_used", False),
                "fallback_reason": call_res.get("fallback_reason", ""),
                "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
                "timing_ms": {"prep": 0, "llm": int((t1 - t0) * 1000), "total": int((t1 - t0) * 1000)},
                "tokens": {
                    "approx_input_tokens": estimate_tokens(" ".join([metadata["title"], metadata["abstract"], metadata["authors"]])),
                    "max_ctx": model_max_ctx(used_model),
                    "prompt_budget": model_max_ctx(used_model) - output_reserve(Mode(mode)) - instruction_reserve(),
                    "tokens_reserved_output": output_reserve(Mode(mode)),
                },
            }
            return str(call_res["answer"]), _format_run_report(meta)

        if paper:
            t0 = time.perf_counter()
            prep = prepare_context_from_paper(paper, question, model_id=model_id, mode=Mode(mode))
            t1 = time.perf_counter()
            call_res = _run_with_timeout(_call_ask_question, timeout, prep["snippets_context"], prep["augmented_question"], model_id, mode)
            t2 = time.perf_counter()
            used_model = call_res.get("chosen_model", model_id)
            meta = {
                "path": "paper(ui)", "requested_model": model_id, "model_id": used_model, "mode": mode,
                "fallback_used": call_res.get("fallback_used", False),
                "fallback_reason": call_res.get("fallback_reason", ""),
                "request_timeout_ms_used": timeout, "timeout_seconds_used": round(timeout / 1000, 3),
                "timing_ms": {"prep": int((t1 - t0) * 1000), "llm": int((t2 - t1) * 1000), "total": int((t2 - t0) * 1000)},
                "tokens": prep["diagnostics"],
            }
            return str(call_res["answer"]), _format_run_report(meta)

        return "Please provide either arXiv metadata (title) or paper text.", ""
    except HTTPException as e:
        return f"[Input error: {str(e.detail)}]", ""
    except Exception as e:
        return f"[Error: {str(e)[:200]}]", ""

def build_help_markdown_static_wrapper():
    return build_help_markdown_static()

# NEW: helpers that *show* the panel when clicked
def _show_help():
    return gr.update(value=build_help_markdown_static(), visible=True)

def _show_latency_selected(model_id, wait_ms):
    return gr.update(value=build_help_with_latency(model_id, wait_ms), visible=True)

def _show_latency_all(wait_ms):
    return gr.update(value=build_help_with_all_latencies(wait_ms), visible=True)

def build_ui():
    with gr.Blocks(title="Query LLM") as demo:
        gr.Markdown("# Query LLM")

        with gr.Row():
            mode = gr.Radio(["short", "detailed"], value="short", label="Answer length", scale=1)

            model_id = gr.Dropdown(MODEL_IDS, value=ModelName.mistral_v02.value, label="Choose the LLM", scale=1)
            request_timeout_ms = gr.Slider(5000, 900000, value=REQUEST_TIMEOUT_MS, step=500, label="Timeout (ms)", scale=2)
            latency_timeout_ms = gr.Slider(5000, LATENCY_MAX_MS, value=LATENCY_DEFAULT_MS, step=5000, label="Latency check (ms)", scale=2)


        with gr.Row():
            help_btn = gr.Button("HELP", variant="secondary", scale=1)
            latency_sel_btn = gr.Button("Refresh latency (selected)", variant="secondary", scale=1)
            latency_all_btn = gr.Button("Refresh latency (all)", variant="secondary", scale=1)


        help_output = gr.Markdown(value="", visible=False)

        with gr.Tab("arXiv metadata"):
            title = gr.Textbox(label="Title")
            abstract = gr.Textbox(label="Abstract", lines=6)
            authors = gr.Textbox(label="Authors")

        with gr.Tab("Paper text"):
            paper = gr.Textbox(label="Paper text", lines=12)
            file_upl = gr.File(label="Upload .txt file", file_types=[".txt"], file_count="single", type="filepath")

            def _on_file_change(path):
                if not path:
                    return ""
                try:
                    if os.path.getsize(path) > FILE_MAX_MB * 1024 * 1024:
                        return f"[Error reading file: file too large > {FILE_MAX_MB} MB]"
                except Exception:
                    pass
                return _load_file_to_text(path)

            file_upl.change(fn=_on_file_change, inputs=file_upl, outputs=paper)

        question = gr.Textbox(label="Your question", lines=3)

        token_info = gr.Markdown(value="Token info will appear here…", visible=True)
        run_report = gr.Markdown(value="", visible=True)

        recalc_btn = gr.Button("Update token info", variant="secondary")
        recalc_btn.click(fn=_token_info_for_paper, inputs=[paper, question, model_id, mode], outputs=[token_info])

        out = gr.Textbox(label="Answer", lines=10, interactive=False, show_copy_button=True,
                         placeholder="Answer will appear here...")

        gr.Button("Ask").click(
            fn=ui_submit,
            inputs=[title, abstract, authors, paper, question, mode, model_id, request_timeout_ms],
            outputs=[out, run_report],
        )


        help_btn.click(fn=_show_help, inputs=None, outputs=[help_output])
        latency_sel_btn.click(fn=_show_latency_selected, inputs=[model_id, latency_timeout_ms], outputs=[help_output])
        latency_all_btn.click(fn=_show_latency_all, inputs=[latency_timeout_ms], outputs=[help_output])

    return demo

def _load_file_to_text(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        size_bytes = os.path.getsize(path)
        if size_bytes > FILE_MAX_MB * 1024 * 1024:
            return f"[Error reading file: file too large > {FILE_MAX_MB} MB]"
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file: {e}]"

# Build and mount UI
demo = build_ui()
app = mount_gradio_app(app, demo, path="/ui")
