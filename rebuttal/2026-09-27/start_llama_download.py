"""Use the author's updated local HF login to download on the authorized server."""
import json
import subprocess
import sys
import time
import getpass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata

token_path=Path.home()/'.cache/huggingface/token'
url=hf_hub_url('meta-llama/Llama-3.1-8B-Instruct','config.json',
              revision='0e9e39f249a16976918f6564b8830bc894c89659',endpoint='https://huggingface.co')
seen=None
prompt_mode='--prompt' in sys.argv
while True:
    current=getpass.getpass('HF token (not saved): ') if prompt_mode else (token_path.read_text().strip() if token_path.exists() else '')
    if current and current!=seen:
        seen=current
        try:
            HfApi(token=current,endpoint='https://huggingface.co').whoami()
            get_hf_file_metadata(url,token=current,timeout=15)
            print('Credential is valid and has access to the pinned Llama model',flush=True)
            break
        except Exception as e:
            print('Waiting for an updated authorized login; status',getattr(getattr(e,'response',None),'status_code',None),flush=True)
            if prompt_mode:
                raise SystemExit(1)
    time.sleep(15)

command=['ssh','-p','20886',
         '-o','ControlPath=/tmp/cass-rebuttal-own-20886',
         '-o','ControlMaster=no',
         '-o','ConnectTimeout=15','-o','BatchMode=yes','root@connect.bjb1.seetacloud.com',
         'source /etc/network_turbo >/dev/null && exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python /root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/download_llama_remote.py > /root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/llama_download.log 2>&1']
for attempt in range(3):
    result=subprocess.run(command,input=json.dumps(dict(token=current)),text=True,check=False)
    if result.returncode==0:
        break
    print('Download connection ended with status',result.returncode,'; retry',attempt+1,flush=True)
    time.sleep(5)
raise SystemExit(result.returncode)
