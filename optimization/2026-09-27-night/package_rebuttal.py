"""Package author-facing response/evidence documents; raw runs stay in the repo."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DOCS=ROOT/'rebuttal/2026-09-27-optimized'


def main():
    files=sorted(p for p in DOCS.iterdir() if p.is_file() and p.suffix in ['.md','.tex','.pdf'])
    files+=sorted(p for p in (HERE/'figures').iterdir() if p.is_file() and p.suffix in ['.pdf','.svg','.png','.csv','.json'])
    manifest=[]
    for p in files:
        raw=p.read_bytes()
        if p.suffix in ['.md','.tex','.svg','.csv','.json']:
            assert not re.search(rb'hf_[A-Za-z0-9]{25,}',raw), ('credential pattern in package',p.name)
        manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    data=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Author-facing response, proof, figure and evidence-document bundle. Includes author notes; inspect individual files before any external submission. Raw generations, activation caches and model weights remain outside this reading bundle.',
        files=manifest)
    out=ROOT/'rebuttal/CASS_rebuttal_author_materials_20260928.zip'
    intro='''# CASS rebuttal author materials

The four responses, compact cajh alternative, technical proof PDF/TeX and
evidence reports are under `rebuttal/2026-09-27-optimized/`.
The same directory contains author notes and proposed manuscript text.
Publication figures (PDF/SVG/PNG) and their numerical CSVs are under
`optimization/2026-09-27-night/figures/`.

This is a reading and editing bundle, not the complete reproduction archive.
Raw generations, experiment scripts, source snapshots, model hashes and
activation caches remain in the shared CASS project. Nothing has been posted
to OpenReview. `package_manifest.json` records the bundled file hashes.
'''
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('README.md',intro)
        z.writestr('package_manifest.json',json.dumps(data,indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for r in manifest:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    receipt=dict(path=str(out.relative_to(ROOT)),files=len(files),bytes=out.stat().st_size,
                 sha256=hashlib.sha256(out.read_bytes()).hexdigest(),member_hashes_verified=True)
    (HERE/'author_package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
