"""Checkpoint only our original broader job after answer-stop verification."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
verification=json.loads((HERE/'answer_stop_verified.json').read_text())
assert len(verification['matched_generations'])==12
state=json.loads((HERE/'worker_gpu2.json').read_text());active=state['active']
assert active['id']=='broader_resume',active
pid=active['pid'];worker=int((HERE/'worker_gpu2.pid').read_text().strip())
assert str(ROOT)==os.readlink(f'/proc/{pid}/cwd')
assert b'broader_run.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'job_worker.py' in Path(f'/proc/{worker}/cmdline').read_bytes()
env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
assert b'CUDA_VISIBLE_DEVICES=2' in env
queue=json.loads((HERE/'queue_gpu2.json').read_text())
assert any(j['id']=='broader_resume_answerstop' for j in queue)
path=HERE/'broader_llama31-8b.jsonl'
os.kill(worker,signal.SIGSTOP)
try:
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Broader child did not stop')
    content=path.read_bytes();lines=content.splitlines(keepends=True);keep=[];partial=None
    for i,line in enumerate(lines):
        try:json.loads(line)
        except json.JSONDecodeError:
            assert i==len(lines)-1;partial=line;break
        keep.append(line.rstrip(b'\n')+b'\n')
    repaired=b''.join(keep)
    if repaired!=content:
        archive=HERE/'archive';archive.mkdir(exist_ok=True)
        (archive/'broader_before_answerstop_checkpoint.jsonl').write_bytes(content)
        path.write_bytes(repaired)
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=pid,worker_pid=worker,
                 complete_records=len(keep),incomplete_trailing_bytes=len(partial or b''),
                 before_sha256=hashlib.sha256(content).hexdigest(),after_sha256=hashlib.sha256(repaired).hexdigest(),
                 verification='answer_stop_verified.json',reason='Resume with verified first-answer stopping; preserve every complete record')
    (HERE/'broader_answerstop_checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
finally:os.kill(worker,signal.SIGCONT)
