"""Archive the exact source/configuration state before an authorized GPU job."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sources=sorted([*HERE.glob('*.py'),*HERE.glob('*.sh'),*HERE.glob('*_selected.json'),
                *ROOT.glob('src/cass/*.py')])
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
if sys.argv[1:]==['--check']:
    print('Manifest source inventory checked:',len(hashes),'files')
    raise SystemExit(0)
gpu=sys.argv[1];assert gpu in ['2','3'] and os.environ['CUDA_VISIBLE_DEVICES']==gpu
entry=Path(sys.argv[2]).resolve();assert entry.parent==HERE and entry.exists()
now=datetime.datetime.now(datetime.timezone.utc)
tag=now.strftime('%Y%m%dT%H%M%S.%fZ')+'_gpu'+gpu+'_'+entry.stem
dest=HERE/'run_manifests';dest.mkdir(exist_ok=True)
archive=dest/(tag+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in sources:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
record=dict(utc=now.isoformat(),gpu=gpu,job_pid=os.getppid(),entry=str(entry.relative_to(ROOT)),args=sys.argv[3:],
            python=sys.version,source_hashes=hashes,archive=archive.name,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            versions={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','scipy']},
            environment={name:os.environ.get(name) for name in ['CUDA_VISIBLE_DEVICES','CASS_MODELS_DIR',
                'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']},
            scope='Source/config snapshot at job launch; model checksums and raw predictions are separate artifacts.')
(dest/(tag+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print('Run source manifest:',tag,flush=True)
