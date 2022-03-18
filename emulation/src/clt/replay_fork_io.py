#!/usr/bin/env python3

### This script replays requests based on a wikipedia log ###
### The log is assumed to preprocessed (by prettify.sh), such that
### each line corresponds to a request
### Input is read from stdin
###
### example format:
### 1190146243.341 /wiki/Germany
import resource
import sys
import time
import math
import random
import socket
import struct
import http.client
import urllib.parse
import os
import threading


def increaseResources():
  # Increase max number of open files (requires root)
  resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))
  # Drop privileges
  uid, gid = os.getenv('SUDO_UID'), os.getenv('SUDO_GID')
  if uid and gid:
    os.setgid(int(gid))
    os.setuid(int(uid))

# IPv6 addresses corresponding to the VIP
addr = '[dead::cafe]'
# Times at which failures happened
failures = []

def runQuery(URL, startTime):
  IP = addr
  c = http.client.HTTPConnection(IP)
  try:
    c.connect()
    s = c.sock
    # Send RST after FIN, to prevent FIN_WAIT from lingering
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
    c.request('GET', URL)
    r = c.getresponse()
  except Exception as e: 
    sys.stdout.write("%f %s failed %s\n" % (startTime - t0, URL, e))
    return
  # Follow one HTTP redirect
  if r.status == 302:
    # r.read()
    try:
      c.request('GET', r.getheader('Location'))
      c.getresponse()
    except Exception as e: 
      sys.stdout.write("%f %s failed %s\n" % (startTime - t0, URL, e))
      return       
  if r.status == 404:
    sys.stdout.write("%f %s failed %s\n" % (startTime - t0, URL, 'HTTP_404'))
    s.close()
    return
  # lines = r.read().decode('utf-8')
  lines = r.read()
  sys.stdout.write("%f %s %s\n" % (startTime - t0, URL, time.time() - startTime))
  s.close()
  return


if __name__ == '__main__':
  if len(sys.argv) <= 1:
    print ("Usage: %s <filename>" % sys.argv[0])
    exit(-1)
  filename = sys.argv[1]
  increaseResources()

  #number of seconds elapsed since the beginning of the replay
  elapsedSeconds = -1

  #list containing (startingTime, url) tuples, corresponding to
  #queries soon to be run
  #invariant: sorted by startingTime
  scheduled = []

  t0 = time.time()

  ## fork n times in order to be able to perform more requests ##
  numProcesses = 8
  isChild = 0
  for i in range(1, numProcesses):
    if isChild == 0:
      isChild = i*int(os.fork() == 0)

  ## open file after fork to have two different copies ##
  file = open(filename, 'r')
  # skip header
  lineNo = 1
  file.readline()
  t0log = float(file.readline().split()[0])
  while True:
    t = time.time()

    ## When we start a new second, read the corresponding second in the log and schedule appropriate queries
    if t > t0 + elapsedSeconds + 0.1:
      elapsedSeconds += 0.1
      while True:
        lineNo += 1
        try: line = file.readline()
        except: continue
        if not line:
          break        
        if lineNo % numProcesses != isChild:
          continue
        try:
          #retrieve the queries to run between t0+elapsedSeconds and t0+elapsedSeconds+1
          line = line.split()
          nextTime = t0 + float(line[0]) - t0log
          url = line[1]
        except: continue
        scheduled.append((nextTime, url))
        if nextTime > t + 0.1:
          break
      if not line and not scheduled:
        break

    ## Every msec, run scheduled queries
    while not not scheduled:
      timeToRun, url = scheduled[0]
      if timeToRun < t:
        th = threading.Thread(target=runQuery, args=[url, t])
        try: th.start()
        except:
          sys.stderr.write('Cannot start thread, sleeping for 1s')
          time.sleep(1)
        scheduled.pop(0)
      else:
        break

    
    dt = 0.001 - (time.time() - t)
    if dt < 0: 
      dt = 0
    time.sleep(dt) # sleeping for 0 will at least yield

  for th in threading.enumerate():
    if th != threading.currentThread():
      th.join()
