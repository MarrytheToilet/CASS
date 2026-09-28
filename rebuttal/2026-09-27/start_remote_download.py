"""Start the public Qwen download on the authorized server; no token is needed."""
import subprocess
command = [
    "ssh","-p","20886",
    "-o","ProxyCommand=nc -X connect -x 127.0.0.1:7897 connect.bjb1.seetacloud.com 20886",
    "-o","KexAlgorithms=curve25519-sha256","-o","ControlMaster=no","-o","ControlPath=none",
    "-o","ConnectTimeout=15","-o","BatchMode=yes",
    "root@connect.bjb1.seetacloud.com",
    "source /etc/network_turbo >/dev/null && exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python /root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/download_models_remote.py > /root/autodl-tmp/CASS-rebuttal-20260927/rebuttal/2026-09-27/model_download.log 2>&1",
]
print("Starting pinned public model download on the authorized server.",flush=True)
completed = subprocess.run(command,check=False)
raise SystemExit(completed.returncode)
