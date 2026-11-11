"""One reviewer pass, following perform_review with AAAI-style settings.

Call 1 writes the review; call 2 is one self-critique round (perform_review with num_reflections=2) that sees
the whole conversation and returns a revised review. No ensemble, no few-shot example. Reasoning is on for
both calls. Returns the parsed review plus everything needed to inspect the run.
"""
import json
import os
import re
import time
from pathlib import Path

import requests
from tokenizers import Tokenizer

from reviewer import prompts

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = "deepseek-v4-flash-0731"
CONTEXT = 131072
SETTINGS = {
    "temperature": 1.0, "top_p": 0.95, "max_tokens": 32768,
    "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},  # "max" = the real "high" on 0731
}
# vLLM's usage doesn't split reasoning from answer tokens, so count the answer text ourselves
# (DeepSeek-V3 tokenizer; V4's may differ slightly).
_TOK = Tokenizer.from_file(str(Path(__file__).resolve().parent.parent / "data/tokenizers/deepseek-v3_tokenizer.json"))
EMPTY_RETRIES = 3
MAX_PAPER_TOKENS = 75000  # papers longer than this (V3 tokens, text + figure descriptions) are dropped
REQUIRED = ["Summary", "Strengths", "Weaknesses", "Soundness", "Presentation", "Contribution",
            "Overall", "Confidence", "Decision"]


def extract_json(output):
    """perform_review's extract_json_between_markers: first ```json block that parses, else any {...}."""
    matches = re.findall(r"```json(.*?)```", output, re.DOTALL) or re.findall(r"\{.*?\}", output, re.DOTALL)
    for s in matches:
        for candidate in (s.strip(), re.sub(r"[\x00-\x1F\x7F]", "", s.strip())):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def chat(messages, seed):
    """One completion. If the answer comes back empty, retry with the next seed (up to EMPTY_RETRIES times).

    Empty answers happen when the model writes its answer inside the thinking block and never closes it, so
    vLLM's reasoning parser files the whole answer under reasoning (~6% of self-critique calls). A new seed
    gives a fresh sample; the same seed would likely repeat the same output.
    """
    t = time.time()
    for attempt in range(EMPTY_RETRIES + 1):
        r = requests.post(f"{BASE_URL}/chat/completions", timeout=7200,
                          json={"model": MODEL, "messages": messages, "seed": seed + attempt, **SETTINGS})
        r.raise_for_status()
        body = r.json()
        msg = body["choices"][0]["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        content = msg.get("content") or ""
        if content:
            break
    return {"content": content, "reasoning": reasoning,
            "answer_tokens": len(_TOK.encode(content).ids), "empty_retries": attempt,
            "finish_reason": body["choices"][0].get("finish_reason"), "usage": body.get("usage", {}),
            "seconds": round(time.time() - t, 1)}


def review(text, system_prompt=prompts.SYSTEM_PROMPT, review_form=prompts.REVIEW_FORM,
           reflection_prompt=prompts.REFLECTION_PROMPT, seed=42):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": review_form + prompts.PAPER_TEMPLATE.format(text=text)}]
    first = chat(messages, seed)
    messages += [{"role": "assistant", "content": first["content"]},
                 {"role": "user", "content": reflection_prompt}]
    second = chat(messages, seed)
    final = extract_json(second["content"])  # a malformed final review scores 0; no fallback
    missing = [k for k in REQUIRED if not final or k not in final]
    return {"review": final, "initial_review": extract_json(first["content"]), "missing_fields": missing,
            "calls": [first, second]}
