"""Minimal client for the OpenAI-compatible API configured in ``.env``.

Used only by data synthesis (script 07) and LLM-judge failure analysis
(script 14); the CASS method itself never calls the API. Copy
``.env.example`` to ``.env`` and fill in ``API_BASE_URL`` / ``API_KEY`` /
``API_MODELS`` (a comma-separated model list)."""
import json
import time
import urllib.request

from .config import ROOT


def _load_env():
    """Parse ``.env`` as ``KEY=VALUE`` lines (blanks and ``#`` comments
    ignored). Returns {} if the file is absent, so importing this module
    never fails on a machine without API credentials."""
    path = ROOT / ".env"
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_env = _load_env()
BASE = _env.get("API_BASE_URL", "").rstrip("/")
KEY = _env.get("API_KEY", "")
MODELS = [m.strip() for m in _env.get("API_MODELS", "").split(",") if m.strip()]


def _require_config():
    if not (BASE and KEY and MODELS):
        raise RuntimeError(
            "API not configured: copy .env.example to .env and set "
            "API_BASE_URL / API_KEY / API_MODELS "
            "(needed only for scripts 07 and 14).")


def chat(prompt, model=None, max_tokens=2000, temperature=0.7, retries=3,
         system=None):
    _require_config()
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
