"""Download authorized, signed HF CDN objects with resumable ranged requests.

Signed URLs arrive over SSH stdin, never enter command arguments or files, and
are used only for HF-owned CDN hosts. Each final file must match its SHA-256.
"""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'models/Llama-3.1-8B-Instruct'
PARTS=ROOT/'llama-cdn-parts'
PARTS.mkdir(exist_ok=True)
BLOCK=32*1024*1024
entries=json.load(sys.stdin)['files']
start=time.monotonic()


def checksum(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()


def get_part(job):
    entry,index,lo,hi=job
    path=PARTS/f"{entry['name']}.{index:04d}.part"
    if path.exists() and path.stat().st_size==hi-lo+1:
        return 0
    for attempt in range(5):
        session=requests.Session(); session.trust_env=False
        session.verify='/etc/ssl/certs/ca-certificates.crt'
        try:
            with session.get(entry['url'],headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(15,30)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range')==f"bytes {lo}-{hi}/{entry['size']}", 'invalid ranged response'
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    for b in r.iter_content(256*1024): f.write(b)
                assert tmp.stat().st_size==hi-lo+1, 'incomplete part'
                tmp.replace(path)
            return hi-lo+1
        except Exception as exc:
            print('Part retry',entry['name'],index,attempt+1,type(exc).__name__,flush=True)
            if attempt==4: raise RuntimeError('Part download exhausted retries') from None
            time.sleep(1+attempt)
        finally: session.close()


jobs=[]
for entry in entries:
    assert urlparse(entry['url']).hostname.endswith('.hf.co')
    assert entry['name'] in ['model-00003-of-00004.safetensors','model-00004-of-00004.safetensors']
    p=DEST/entry['name']
    if p.exists() and p.stat().st_size==entry['size'] and checksum(p)==entry['sha256']:
        continue
    for i,lo in enumerate(range(0,entry['size'],BLOCK)):
        jobs.append((entry,i,lo,min(lo+BLOCK,entry['size'])-1))
total=0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures=[pool.submit(get_part,j) for j in jobs]
    for i,f in enumerate(concurrent.futures.as_completed(futures),1):
        total+=f.result()
        if i%4==0 or i==len(futures):
            print('Completed parts',i,'/',len(futures),'received MB',round(total/1e6,1),'MB/s',round(total/(time.monotonic()-start)/1e6,2),flush=True)
for entry in entries:
    p=DEST/entry['name']
    if p.exists() and p.stat().st_size==entry['size'] and checksum(p)==entry['sha256']:
        continue
    tmp=p.with_suffix('.assembling')
    with tmp.open('wb') as f:
        for part in sorted(PARTS.glob(entry['name']+'.*.part')):
            with part.open('rb') as g:
                while b:=g.read(8*1024*1024): f.write(b)
    assert tmp.stat().st_size==entry['size'] and checksum(tmp)==entry['sha256']
    tmp.replace(p)
    print('SHA-256 verified',entry['name'],flush=True)
receipt=dict(source='Authorized signed URLs obtained from official HF API; ranged CDN download',
             revision='0e9e39f249a16976918f6564b8830bc894c89659',
             files=[{k:v for k,v in e.items() if k!='url'} for e in entries])
(ROOT/'rebuttal/2026-09-27/llama_transfer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Llama ready for full checkpoint verification',flush=True)
