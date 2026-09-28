"""Obtain authorized HF URLs locally, then send them transiently to the server."""
import getpass
import json
import subprocess
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url

token=getpass.getpass('HF token (not saved): ')
revision='0e9e39f249a16976918f6564b8830bc894c89659'
original=json.loads((Path(__file__).parent/'original_model_hashes.json').read_text())['Llama-3.1-8B-Instruct']
files=[]
for name in ['model-00003-of-00004.safetensors','model-00004-of-00004.safetensors']:
    m=get_hf_file_metadata(hf_hub_url('meta-llama/Llama-3.1-8B-Instruct',name,revision=revision,endpoint='https://huggingface.co'),token=token,timeout=30)
    assert m.size==original[name]['size'] and m.etag==original[name]['sha256']
    files.append(dict(name=name,url=m.location,size=m.size,sha256=m.etag))
    print('Authorized matching original shard',name,flush=True)
del token
command=['ssh','-o','ControlPath=/tmp/cass-rebuttal-own-20886','-o','ControlMaster=no',
         '-p','20886','root@connect.bjb1.seetacloud.com',
         'exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python /root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/download_cdn_chunks.py > /root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/cdn_download.log 2>&1']
p=subprocess.run(command,input=json.dumps(dict(files=files)),text=True)
raise SystemExit(p.returncode)
