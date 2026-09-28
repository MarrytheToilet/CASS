"""Fetch the pinned Llama checkpoint with an explicitly supplied valid token.

Authentication arrives through SSH stdin and is used only for the official
huggingface.co endpoint. It is not logged or saved. AutoDL proxy is shell-scoped.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HOME']=str(ROOT/'.hf-cache')
os.environ['HF_ENDPOINT']='https://huggingface.co'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
# The AutoDL proxy covers the HF API host. Download CDN hosts use a direct
# connection; hf_transfer's proxy handling failed before writing any weights.
os.environ['no_proxy']=os.environ.get('no_proxy','')+',.hf.co,.xethub.hf.co'
os.environ['NO_PROXY']=os.environ['no_proxy']
os.environ['HF_HUB_ETAG_TIMEOUT']='30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import snapshot_download, get_hf_file_metadata, hf_hub_url
import requests

token=json.load(sys.stdin)['token']
revision='0e9e39f249a16976918f6564b8830bc894c89659'
dest=ROOT/'models/Llama-3.1-8B-Instruct'
meta=get_hf_file_metadata(hf_hub_url('meta-llama/Llama-3.1-8B-Instruct',
    'model-00003-of-00004.safetensors',revision=revision),token=token)
proxy=os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
rates={}
for route in ['proxy','direct']:
    session=requests.Session(); session.trust_env=False
    session.verify=os.environ.get('REQUESTS_CA_BUNDLE',True)
    if route=='proxy': session.proxies={'https':proxy,'http':proxy}
    try:
        start=time.perf_counter(); received=0
        with session.get(meta.location,headers={'Range':'bytes=0-4194303'},stream=True,timeout=(10,10)) as response:
            response.raise_for_status()
            for block in response.iter_content(256*1024):
                received+=len(block)
                if received>=4194304: break
        rates[route]=received/(time.perf_counter()-start)
        print('CDN route probe',route,'MB/s',round(rates[route]/1e6,3),flush=True)
    except Exception as exc:
        print('CDN route probe',route,'failed',type(exc).__name__,flush=True)
    finally: session.close()
if rates.get('proxy',0)>rates.get('direct',0):
    os.environ['no_proxy']=','.join(x for x in os.environ['no_proxy'].split(',') if x not in ['.hf.co','.xethub.hf.co'])
    os.environ['NO_PROXY']=os.environ['no_proxy']
    print('Using AutoDL proxy for API and CDN',flush=True)
else:
    print('Using AutoDL proxy for API and direct CDN route',flush=True)
print('Downloading authorized pinned Llama checkpoint through AutoDL proxy',flush=True)
snapshot_download(repo_id='meta-llama/Llama-3.1-8B-Instruct',revision=revision,
                  token=token,local_dir=dest,max_workers=4,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE'])
(ROOT/'rebuttal/2026-09-27/llama_transfer_receipt.json').write_text(json.dumps(
    dict(source='official HF through AutoDL proxy',revision=revision,local_path=str(dest)),indent=2)+'\n')
print('Llama model ready; hash verification follows',flush=True)
