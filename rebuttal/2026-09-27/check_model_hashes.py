"""Record local original model hashes, or validate the isolated server copies."""
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
NAMES=['Llama-3.1-8B-Instruct','Qwen3-4B']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()


if sys.argv[1]=='record':
    data={}
    for name in NAMES:
        root=Path('/home/hanyu/models')/name
        data[name]={}
        for p in sorted(root.iterdir()):
            if p.suffix in ['.safetensors','.json','.txt'] and p.is_file():
                data[name][p.name]=dict(size=p.stat().st_size,sha256=sha(p))
                print('Hashed',name,p.name,flush=True)
    (HERE/'original_model_hashes.json').write_text(json.dumps(data,indent=2)+'\n')
else:
    name=sys.argv[1]
    assert name in NAMES
    expected=json.loads((HERE/'original_model_hashes.json').read_text())[name]
    root=HERE.parents[1]/'models'/name
    for filename,meta in expected.items():
        p=root/filename
        assert p.stat().st_size==meta['size'],filename
        assert sha(p)==meta['sha256'],filename
    print('Verified original checkpoint',name,flush=True)
    (HERE/(name+'_hash_verified.json')).write_text(json.dumps(dict(model=name,matched_local_original=True,files=expected),indent=2)+'\n')
