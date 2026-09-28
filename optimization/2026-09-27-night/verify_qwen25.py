import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
manifest=json.loads((here/'qwen25_expected_hashes.json').read_text())
folder=here.parents[1]/'models'/'Qwen2.5-3B-Instruct'
checks={}
for name,expected in manifest['files'].items():
 p=folder/name
 assert p.stat().st_size==expected['size'],name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 assert h.hexdigest()==expected['sha256'],name
 checks[name]=True
(here/'qwen25_verified.json').write_text(json.dumps(dict(revision=manifest['revision'],matched_local_original=checks),indent=2)+'\n')
print('Verified all',len(checks),'checkpoint files against the original local model.',flush=True)
