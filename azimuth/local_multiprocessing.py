import multiprocessing
import numpy as np


def configure(num_jobs=8, TEST=False, subtract=0, num_proc=None, num_thread_per_proc=None):
    '''
    num_jobs is typically the # of genes we are parallelizing over
    '''
    if num_proc is None:
        num_proc = multiprocessing.cpu_count() - subtract

    if num_jobs > num_proc:
        num_jobs = num_proc

    if num_thread_per_proc is None:
        num_thread_per_proc = int(np.floor(num_proc/num_jobs))

    if TEST:
        num_jobs = 1
        num_thread_per_proc = 1

    try:
        import mkl
        mkl.set_num_threads(num_thread_per_proc)    
    except ImportError:
        print "MKL not available, so I'm not adjusting the number of threads"

    print "Launching %d jobs with %d MKL threads each" % (num_jobs, num_thread_per_proc)

    return num_jobs
