import os, select, socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1000)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
sock.settimeout(20)
sock.connect((sys.argv[1], int(sys.argv[2])))
sock.settimeout(None)
stdin_open = True
while True:
    ready, _, _ = select.select(([0] if stdin_open else []) + [sock], [], [])
    if sock in ready:
        data = sock.recv(65536)
        if not data: break
        view = memoryview(data)
        while view:
            view = view[os.write(1, view):]
    if 0 in ready:
        data = os.read(0, 1000)
        if not data:
            stdin_open = False
            sock.shutdown(socket.SHUT_WR)
        else:
            sock.sendall(data)
