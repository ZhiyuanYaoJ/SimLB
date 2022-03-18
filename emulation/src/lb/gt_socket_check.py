import socket
import sys

'''
check ground truth gathering socket connection
'''
HOST = ['10.0.1.{}'.format(i) for i in range(1, int(sys.argv[1]) + 1)]   # The list of remote hosts
PORT = 50008              # The same port as used by the server
def get_err_sockets(host_list, port):
    err = []
    for host in host_list:
        for res in socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
            s = None  
            af, socktype, proto, canonname, sa = res
            try:
                s = socket.socket(af, socktype, proto)
            except OSError as msg:
                s = None
                continue
            try:
                s.connect(sa)
            except OSError as msg:
                s.close()
                s = None
                continue
            break
        if s is None:
            err.append(int(host.split('.')[-1]) - 1)
            # print('{} could not open socket'.format(host))
        else:
            s.close()
    return err
print(get_err_sockets(HOST, PORT))
# print("GT socket check pass.")