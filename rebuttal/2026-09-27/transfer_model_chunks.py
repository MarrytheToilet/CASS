"""Copy existing author checkpoint shards over authenticated SSH in bounded chunks.

Only project-local chunk files are written. Completed chunks resume by size;
the final model files are SHA-256 checked against the local originals.
"""
import concurrent.futures
import hashlib
import json
import math
import shlex
import subprocess
import time
from pathlib import Path

SOURCE=Path('/home/hanyu/models/Llama-3.1-8B-Instruct')
DEST='/root/autodl-tmp/CASS-rebuttal-20260927/models/Llama-3.1-8B-Instruct'
CHUNKS='/root/autodl-tmp/CASS-rebuttal-20260927/model-transfer-chunks-4m'
SSH=['ssh','-p','20886','-o','HostName=106.120.183.117',
     '-o','HostKeyAlias=[connect.bjb1.seetacloud.com]:20886',
     '-o','ProxyCommand=/home/hanyu/miniconda3/bin/python /home/hanyu/research/CASS/rebuttal/2026-09-27/ssh_transport.py %h %p',
     '-o','IPQoS=none','-o','Ciphers=aes128-ctr','-o','Compression=yes',
     '-o','KexAlgorithms=curve25519-sha256',
     '-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15',
     '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
     '-o','BatchMode=yes','root@connect.bjb1.seetacloud.com']
BLOCK=4*1024*1024
FILES=['model-00003-of-00004.safetensors','model-00004-of-00004.safetensors']


def remote(code,data=None,timeout=120):
    command=SSH+['/root/miniconda3/envs/dtr/bin/python -c '+shlex.quote(code)]
    p=subprocess.run(command,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-1500:])
    return p.stdout


def main():
    existing=json.loads(remote(f'import pathlib,json; p=pathlib.Path({CHUNKS!r}); p.mkdir(exist_ok=True); print(json.dumps({{f.name:f.stat().st_size for f in p.glob("*.part")}}))'))
    work=[]
    for name in FILES:
        size=(SOURCE/name).stat().st_size
        for i in range(math.ceil(size/BLOCK)):
            n=min(BLOCK,size-i*BLOCK)
            key=f'{name}.{i:04d}.part'
            if existing.get(key)!=n:
                work.append((name,i,n,key))
    total=sum(t[2] for t in work); start=time.monotonic(); completed=0
    print('Pending bytes',total,'chunks',len(work),flush=True)
    def send(task):
        name,i,n,key=task
        with (SOURCE/name).open('rb') as f:
            f.seek(i*BLOCK); data=f.read(n)
        assert len(data)==n
        path=f'{CHUNKS}/{key}'
        code=f'import sys,pathlib,hashlib; b=sys.stdin.buffer.read({n}); assert len(b)=={n}; assert hashlib.sha256(b).hexdigest()=={hashlib.sha256(data).hexdigest()!r}; p=pathlib.Path({path!r}); q=p.with_suffix(".tmp"); q.write_bytes(b); q.replace(p)'
        for attempt in range(3):
            try:
                remote(code,data,timeout=240)
                return n,key
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                if attempt==2:
                    raise
                print('Retry',key,type(exc).__name__,str(exc)[-200:],flush=True)
                time.sleep(1+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(send,t) for t in work]
        for f in concurrent.futures.as_completed(futures):
            n,key=f.result(); completed+=n
            elapsed=time.monotonic()-start
            print('Transferred',completed,'/',total,'bytes',round(completed/elapsed/1e6,2),'MB/s',key,flush=True)
    # The source files are read once for the final full-shard integrity check.
    hashes={}
    for name in FILES:
        h=hashlib.sha256()
        with (SOURCE/name).open('rb') as f:
            while block:=f.read(8*1024*1024):
                h.update(block)
        hashes[name]=h.hexdigest()
    code=f'''import pathlib,hashlib,json
root=pathlib.Path({CHUNKS!r}); dest=pathlib.Path({DEST!r}); expected={hashes!r}
for name,digest in expected.items():
 p=dest/(name+'.assembling'); h=hashlib.sha256()
 with p.open('wb') as out:
  for c in sorted(root.glob(name+'.*.part')):
   with c.open('rb') as f:
    while b:=f.read(8*1024*1024): out.write(b); h.update(b)
 assert h.hexdigest()==digest, name
 p.replace(dest/name)
receipt=pathlib.Path('/root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/llama_transfer_receipt.json')
receipt.write_text(json.dumps(dict(source='existing author checkpoint',revision='0e9e39f249a16976918f6564b8830bc894c89659',sha256=expected),indent=2)+'\\n')
print('Llama missing shards assembled and SHA-256 verified')
'''
    print(remote(code,timeout=240).decode(),flush=True)


if __name__=='__main__':
    main()
