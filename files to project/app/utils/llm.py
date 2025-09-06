from __future__ import annotations
import os
from typing import Optional
from transformers import pipeline
from dotenv import load_dotenv
import threading


# ENV 

if os.path.exists(".env"):
    load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None) 
CPU_FALLBACK_MODEL = os.getenv("CPU_FALLBACK_MODEL", "microsoft/Phi-3-mini-4k-instruct")
ALLOW_HEAVY_ON_CPU = os.getenv("ALLOW_HEAVY_ON_CPU", "0") in {"1", "true", "True", "YES", "yes"}


# Hardware detection

def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_HEAVY_7B = {
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "tiiuae/falcon-7b-instruct",
}


_META_MARKERS = (
    "is on the meta device, we need a `value` to put in on",
    "meta device",
    "accelerate/hooks.py",
    "set_module_tensor_to_device",
    "device_map",
    "tensor.item() cannot be called on meta tensors",  
)
def _looks_like_meta_cpu_error(exc: Exception) -> bool:
    msg = f"{exc}".lower()
    return any(m in msg for m in (s.lower() for s in _META_MARKERS))


# Generator cache + locks

_GEN_CACHE = {}          # requested_model -> pipeline instance
_GEN_LOCKS = {}          # requested_model -> threading.Lock
_GEN_LOCKS_GLOBAL = threading.Lock()

def _get_lock(key: str) -> threading.Lock:

    with _GEN_LOCKS_GLOBAL:
        if key not in _GEN_LOCKS:
            _GEN_LOCKS[key] = threading.Lock()
        return _GEN_LOCKS[key]

def _contains_meta_params(m) -> bool:
    try:
        for _, p in m.named_parameters(recurse=True):
            if getattr(p, "is_meta", False):
                return True
        return False
    except Exception:
        return False

def _build_generator_unlocked(requested_model: str):

    import torch

    use_model = requested_model
    kwargs = {}

    if _has_cuda():
  
        kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        })
    else:

        if (requested_model in _HEAVY_7B) and not ALLOW_HEAVY_ON_CPU:
            use_model = CPU_FALLBACK_MODEL
        kwargs.update({
            "device": -1,  
        })

    gen = pipeline(
        task="text-generation",
        model=use_model,
        tokenizer=use_model,
        framework="pt",
        **kwargs
    )


    mdl = getattr(gen, "model", None)
    if mdl is not None and _contains_meta_params(mdl):
        raise RuntimeError(
            f"Model `{use_model}` contains meta tensors after load"
        )

    return gen

def _build_generator(requested_model: str):

    lock = _get_lock(requested_model)
    if requested_model in _GEN_CACHE:
        return _GEN_CACHE[requested_model]
    with lock:

        if requested_model in _GEN_CACHE:
            return _GEN_CACHE[requested_model]
        gen = _build_generator_unlocked(requested_model)
        _GEN_CACHE[requested_model] = gen  
        return gen

def _safe_get_generator(requested_model: str):

    try:
        return _build_generator(requested_model)
    except Exception as e:
        if (not _has_cuda()) and _looks_like_meta_cpu_error(e):
            
            _GEN_CACHE.pop(requested_model, None)
            if requested_model != CPU_FALLBACK_MODEL:
                _GEN_CACHE.pop(CPU_FALLBACK_MODEL, None)
                
                return _build_generator(CPU_FALLBACK_MODEL)

        raise


# Public API 

def get_generator(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):

    return _safe_get_generator(model_name)

def ask_question(context: str, question: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:

    prompt = (
        f"You are a helpful and concise assistant.\n\n"
        f"Context:\n\"\"\"{context}\"\"\"\n\n"
        f"Question:\n\"\"\"{question}\"\"\"\n\nAnswer:"
    )
    try:
        generator = get_generator(model_name)
        result = generator(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
        )
        text = result[0].get("generated_text", "")
        return text.split("Answer:", 1)[-1].strip() if "Answer:" in text else text.strip()
    except Exception as e:

        if (not _has_cuda()) and _looks_like_meta_cpu_error(e) and model_name != CPU_FALLBACK_MODEL:
            try:
                _GEN_CACHE.pop(model_name, None)
                _GEN_CACHE.pop(CPU_FALLBACK_MODEL, None)
                generator = get_generator(CPU_FALLBACK_MODEL)
                result = generator(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
                text = result[0].get("generated_text", "")
                return text.split("Answer:", 1)[-1].strip() if "Answer:" in text else text.strip()
            except Exception as e2:
                return f"Error (fallback failed): {str(e2)}"
        return f"Error: {str(e)}"

def ask_arxiv_question(metadata: dict, question: str, model_key: Optional[str] = None, mode: str = "short") -> str:
    """
    בונה קונטקסט מסיכום המטא־דאטה של מאמר arXiv וקורא ל-ask_question.
    שומר על החתימה שלך: הפרמטר model_key הוא שם המודל שיוזן ל-ask_question.
    """
    title = (metadata or {}).get("title", "")
    abstract = (metadata or {}).get("abstract", "")
    authors = (metadata or {}).get("authors", "")

    if mode == "detailed":
        context = (
            f"Answer the question thoroughly using this arXiv paper.\n\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract: {abstract}"
        )
        aug_q = question + "\nPlease provide a detailed explanation."
    else:
        context = (
            f"Answer the question briefly and concisely using this arXiv paper.\n\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract: {abstract}"
        )
        aug_q = question


    return ask_question(context, aug_q, model_key or "mistralai/Mistral-7B-Instruct-v0.2")
