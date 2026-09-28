"""Content hashes for the saved activation/signature caches and run archives."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['create','verify'])
    args=parser.parse_args()
    manifest=HERE/'cache_artifact_inventory.json'
    if args.action=='create':
        files=sorted([p for p in HERE.rglob('*.pt') if 'inductor_cache' not in p.parts]
                     +list((HERE/'run_manifests').glob('*.tar.gz')))
        rows=[]
        for p in files:
            before=p.stat();digest=sha256(p);after=p.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), ('file changed during hash',p.name)
            rows.append(dict(path=str(p.relative_to(HERE)),bytes=after.st_size,sha256=digest))
        data=dict(files=rows,n_files=len(rows),bytes=sum(r['bytes'] for r in rows),
                  scope='Activation/signature tensors and exact source archives. Model-weight hashes and prediction-source hashes are recorded separately.')
        manifest.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(n_files=data['n_files'],mib=data['bytes']/2**20)))
    else:
        data=json.loads(manifest.read_text());issues=[]
        for row in data['files']:
            p=HERE/row['path']
            if not p.is_file():issues.append(dict(path=row['path'],error='missing'));continue
            if p.stat().st_size!=row['bytes'] or sha256(p)!=row['sha256']:
                issues.append(dict(path=row['path'],error='content mismatch'))
        report=dict(n_expected=data['n_files'],n_verified=data['n_files']-len(issues),issues=issues,
                    manifest_sha256=sha256(manifest))
        (HERE/'cache_artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
        assert not issues


if __name__=='__main__':main()
