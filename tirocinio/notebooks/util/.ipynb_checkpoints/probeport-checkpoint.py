#!/usr/bin/env python3

import os,sys
from socket import *

def probeport(host,port):
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect((host,port))
        print('%s:%d tcp open'%(host,port))
        sock.close()
    except:
        print('%s:%d/tcp closed'%(host,port))
    finally:
        sock.close()

host,port = sys.argv[1],int(sys.argv[2])

x = 0
try:
    x = inet_aton(host)
except:
    pass

ip = ''
if x == 0:
    try:
        ip = gethostbyaddr(host)[-1][0]
    except:
        print("cannot resolve %s"%host)
        sys.exit(1)
else:
    host = ip

probeport(host,port)

