#coding=utf-8
from multiprocessing import Pool
import os, time, random

# The Unstable OpenCV may cause difficulty in generating images.
# Use shell cmd to restart:
# kill $(ps -fe | grep "[p]ython generator_wrapper.py" | awk '{print $2}')
# kill $(ps -fe | grep "[p]ython generator_varlen_5_45_CHLang.py" | awk '{print $2}')

def long_time_task(idx):
    """
    Single Task.
    """
    pid = os.getpid()
    while 1:
        print pid
        #print 'Run task %s (%s)...' % (name, os.getpid())
        start = time.time()
        os.system("nohup python generator_4_CH.py &")
        while time.time() <= start + 80:
            time.sleep(10)
            print "Fresh..."
        os.system("kill $(ps -fe | grep \"[p]ython generator_4_CH.py\" | awk '{print $2}')")
        #os.system("kill $(ps -fe | grep \"[p]ython generator_wrapper.py\" | awk '{print $2}')")
    #print 'Task %s runs %0.2f seconds.' % (name, (end - start))

if __name__=='__main__':
    
    print 'Parent process %s.' % os.getpid()
    p = Pool()
    for i in range(10):
        p.apply_async(long_time_task, args=(i,))
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    print 'All subprocesses done.'


