"""Scoped SSH transport for the authorized server, with a smaller TCP MSS.

This changes only this socket, not host routing or global network settings.
It handles no credentials. Used to diagnose poor bulk transfer through the
local virtual network's advertised 9000-byte MTU.
"""
import os
import socket
import sys
import threading

host, port = sys.argv[1], int(sys.argv[2])
if host not in ["106.120.183.117", "connect.bjb1.seetacloud.com"] or port != 20886:
    raise SystemExit("This transport is restricted to the authorized CASS server")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1000)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.settimeout(15)
s.connect((host, port))
s.settimeout(None)

def upstream():
    try:
        while True:
            data = os.read(0, 1000)
            if not data:
                s.shutdown(socket.SHUT_WR)
                break
            s.sendall(data)
    except OSError:
        pass

threading.Thread(target=upstream, daemon=True).start()
try:
    while True:
        data = s.recv(65536)
        if not data:
            break
        view = memoryview(data)
        while view:
            n = os.write(1, view)
            view = view[n:]
finally:
    s.close()
