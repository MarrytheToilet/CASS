"""Public checkpoint download through the AutoDL-recommended HF mirror."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HOME']=str(ROOT/'.hf-cache')
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HUB_OFFLINE']='0'
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
os.environ['HF_HUB_ETAG_TIMEOUT']='25'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
for name in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy']:
    os.environ.pop(name,None)
from huggingface_hub import snapshot_download

repo='Qwen/Qwen2.5-3B-Instruct'
try:
    revision='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    destination=ROOT/'models'/'Qwen2.5-3B-Instruct'
    print('Downloading public checkpoint',repo,revision,flush=True)
    snapshot_download(repo_id=repo,revision=revision,token=False,local_dir=destination,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE'],max_workers=4)
    (Path(__file__).resolve().parent/'qwen25_download.json').write_text(
        json.dumps(dict(repo=repo,revision=revision,path=str(destination)))+'\n')
    print('Qwen2.5-3B checkpoint ready',flush=True)
except Exception as error:
    # Do not log temporary signed URLs from transfer exceptions.
    print('Download failed:',type(error).__name__,flush=True)
    raise SystemExit(1)
