"""One isolated serial worker per authorized GPU; reads a reviewable JSON queue."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser();parser.add_argument('gpu',type=int,choices=[2,3])
parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
HERE=Path(__file__).resolve().parent
queue=HERE/f'queue_gpu{args.gpu}.json';statepath=HERE/f'worker_gpu{args.gpu}.json'
state=json.loads(statepath.read_text()) if statepath.exists() else {'completed':{},'active':None}
deadline=datetime.datetime(2026,9,28,3,18,42,tzinfo=datetime.timezone.utc).timestamp()


def save():
    state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statepath)


if args.wait_pid:
    state['waiting_for_pid']=args.wait_pid;save()
    while True:
        if time.time()>=deadline:
            state['deadline_reached']=True;save();raise SystemExit(0)
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        stat=Path(f'/proc/{args.wait_pid}/stat')
        if stat.exists() and stat.read_text().split()[2]=='Z':break
        time.sleep(5)
    state.pop('waiting_for_pid',None)

while True:
    if time.time()>=deadline:
        state['deadline_reached']=True;state['active']=None;save();break
    jobs=json.loads(queue.read_text())
    job=next((j for j in jobs if j['id'] not in state['completed']),None)
    if job is None:
        state['active']=None;save();time.sleep(5);continue
    assert '/' not in job['id']
    if job.get('skip_if_exists') and (HERE/job['skip_if_exists']).exists():
        state['completed'][job['id']]={'returncode':0,'already_complete':True};save();continue
    script=HERE/job['script']
    assert script.parent==HERE and script.suffix=='.py' and script.exists()
    state['active']={'id':job['id'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};save()
    command=['bash',str(HERE/'run_gpu.sh'),str(args.gpu),str(script),*job.get('args',[])]
    with (HERE/f"job_{job['id']}.log").open('a') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
        state['active']['pid']=proc.pid;save()
        while True:
            try:code=proc.wait(timeout=5);break
            except subprocess.TimeoutExpired:
                if time.time()>=deadline:
                    proc.terminate()
                    try:code=proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
                    state['deadline_reached']=True
                    break
    state['completed'][job['id']]={'returncode':code,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    state['active']=None;save()
