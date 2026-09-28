"""Fetch the pinned public Qwen revision through AutoDL's recommended HF mirror.

No authentication is sent to the mirror. Llama is supplied separately from the
author's existing local files. Shared source directories are not modified.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT/".hf-cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# AutoDL's official network page also recommends this public HF mirror. Its
# direct route measured faster than the proxy for these public weight files.
for variable in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy']:
    os.environ.pop(variable,None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "20"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

allowed = {
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}
receipt_path = ROOT/"rebuttal/2026-09-27/model_download_receipts.json"
receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
for repo,revision in allowed.items():
    if repo.startswith('meta-llama/'):
        # This account returned 401. Use the author's existing local checkpoint
        # through rsync instead; do not attempt to bypass repository access.
        print('Llama checkpoint will be supplied from existing local author files',flush=True)
        continue
    name = repo.split("/")[-1]
    dest = ROOT/"models"/name
    dest.mkdir(parents=True,exist_ok=True)
    for p in (Path("/root/autodl-tmp/models")/name).glob("*"):
        if p.is_file() and not (dest/p.name).exists():
            (dest/p.name).symlink_to(p)
    print("Downloading pinned model",repo,revision,flush=True)
    snapshot_download(repo_id=repo,revision=revision,token=False,local_dir=dest,
                      allow_patterns=["*.json","*.safetensors","*.txt","*.model","LICENSE"],
                      max_workers=4)
    receipts.append(dict(repo=repo,revision=revision,local_path=str(dest)))
    receipt_path.write_text(json.dumps(receipts,indent=2)+"\n")
    print("Model ready",name,flush=True)
