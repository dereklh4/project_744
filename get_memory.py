import os
import sys
import psutil
import time

def memory_calc(pid):
    while psutil.pid_exists(pid):
        process = psutil.Process(pid)
        mem = process.memory_info().rss
        mem_in_mb = mem / float(2**20)
        print (mem_in_mb)
        cpu_percent = process.cpu_percent(0.1)
        print (cpu_percent)
        print (os.getloadavg())
        time.sleep(1)

if __name__ == '__main__':
    memory_calc(int(sys.argv[1]))
