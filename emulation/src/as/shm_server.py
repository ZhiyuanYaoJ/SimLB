#!/usr/bin/env python3

import mmap
import struct
import time
import socket
import sys
from threading import Thread
import _thread

HOST = None               # Symbolic name meaning all available interfaces
PORT = 50008              # Arbitrary non-privileged port

# init socket
s = None
for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                              socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
    af, socktype, proto, canonname, sa = res
    try:
        s = socket.socket(af, socktype, proto)
    except OSError as msg:
        s = None
        continue
    try:
        s.bind(sa)
        s.listen(1)
    except OSError as msg:
        s.close()
        s = None
        continue
    break
if s is None:
    print('could not open socket')
    sys.exit(1)

def listen(conn, addr):
    print('Connected by', addr)
    while True:
        # decode message written by other process at [42:110)
        data_recv = conn.recv(42)
        if not data_recv: break
        # (cpu, mlog, apache, a_str) = struct.unpack('dqi48s', mem[42:110])
        # data_send = "cpu:{:.4%}, memory:{}, apache:{}, {}".format(cpu, mlog, apache, a_str.decode("utf-8").rstrip("\n")).encode("utf-8")
        # print(data_send)
        # conn.send(data_send)
        # conn.send(mem[42:62]) # [cpu:apache:memory]
        conn.send(mem[42:66]) # [cpu:apache:memory:asid]
        #conn.send(mem[42:234]) # [cpu:apache:memory:asid]

        # # send data to other process at [110:178)
        # msg = struct.pack('II64s', i, i+1, data_recv)
        # mem[110:182] = msg
        # i = i+1
    conn.close()

# init shared memory
i = 0
with open("/dev/shm/shm_vpp", "r+") as f:
  mem = mmap.mmap(f.fileno(), 1048576, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, 0)
  
  while True:
    c, addr = s.accept()
    _thread.start_new_thread(listen, (c, addr))
s.close()
