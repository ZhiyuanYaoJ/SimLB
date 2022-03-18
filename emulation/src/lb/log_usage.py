#!/usr/bin/env python
import psutil
import time

interval = 0.5
filename = 'log/usage.log'
f = open(filename, "w")

t0 = time.time()

# print(','.join(['ts', 'cpu_usage', 'used_ram', 'avail_ram']))
f.write(','.join(['ts', 'cpu_usage', 'used_ram', 'avail_ram'])+'\n')
while True:
    time.sleep(interval)

    cpu_usage = psutil.cpu_percent() # gives a single float value
    used_ram = psutil.virtual_memory().percent
    avail_ram = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

    f.write(','.join([str(time.time()-t0), '{:.3f}'.format(cpu_usage), '{:.3f}'.format(used_ram), '{:.3f}'.format(avail_ram)])+'\n')
    # print(','.join([str(time.time()-t0), '{:.3f}'.format(cpu_usage), '{:.3f}'.format(used_ram), '{:.3f}'.format(avail_ram)]))
    f.flush()

f.close()
