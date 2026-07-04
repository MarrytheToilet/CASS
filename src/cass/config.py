import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("CASS_MODELS_DIR", "/home/hanyu/models"))
DATA_DIR = ROOT / "third_party" / "function_vectors" / "dataset_files"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

MODEL_PATHS = {
    "llama31-8b": MODELS_DIR / "Llama-3.1-8B-Instruct",
    "qwen25-7b": MODELS_DIR / "Qwen2.5-7B-Instruct",
    "mistral-7b": MODELS_DIR / "Mistral-7B-Instruct-v0.3",
    "qwen3-4b": MODELS_DIR / "Qwen3-4B",
    "llama32-3b": MODELS_DIR / "Llama-3.2-3B",
    "gemma2-2b": MODELS_DIR / "gemma-2-2b-it-ms",
    "qwen25-15b": MODELS_DIR / "Qwen2.5-1.5B",
    "qwen25-3b": MODELS_DIR / "Qwen2.5-3B-Instruct",
}

def results_dir(model_key: str) -> Path:
    p = RESULTS_DIR / model_key
    p.mkdir(parents=True, exist_ok=True)
    return p
