"""Minimal client for the OpenAI-compatible API in .env (data synthesis and
LLM-judge diagnostics; the CASS method itself never uses the API)."""
import json
import time
import urllib.request
from pathlib import Path

from .config import ROOT

_lines = open(ROOT / ".env").read().strip().split("\n")
BASE, KEY = _lines[0].strip().rstrip("/"), _lines[1].strip()
MODELS = _lines[2].strip().split(",")


def chat(prompt, model=None, max_tokens=2000, temperature=0.7, retries=3,
         system=None):
    model = model or MODELS[0]
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    body = dict(model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                BASE + "/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def chat_json(prompt, **kw):
    """Chat and parse the first JSON array/object in the reply."""
    txt = chat(prompt, **kw)
    start = min((i for i in (txt.find("["), txt.find("{")) if i >= 0),
                default=-1)
    if start < 0:
        raise ValueError(f"no JSON in reply: {txt[:200]}")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(txt[start:])
    return obj
