import multiprocessing as MP
import sys
import traceback
num_workers = max(1, MP.cpu_count() - 1)
def threadit(func, arglist, OnOffSwitch={'state':False}):
    """
    A wrapper for multi-threading any function (func) given a argument list (arglist). The OnOffSwitch is a flag that got set to True when a progress is already running in a thread. It would not spam more threads when the flag is set to True.
    """
    def worker(q,retq, pipe, func, arglist):
        while True:
            idx = q.get()
            if idx is not None:
                retq.put({idx:func(*(arglist[idx]))})
            else:
                break
            q.task_done()
        q.task_done()
    #print func.__name__, ' OnOffSwitch:', OnOffSwitch['state'], len(arglist)
    if OnOffSwitch['state'] == False or len(arglist) <=3:
        #if no threading is already running or the number of jobs to spaw is smaller than 3, don't thread it.
        OnOffSwitch['state'] = True
        q=MP.JoinableQueue()
        to_child, to_self = MP.Pipe()
        retq=MP.Queue()
        procs = []
        for i in range(num_workers):
            p = MP.Process(target=worker, args=(q, retq, to_self, func, arglist))
            p.daemon = True
            p.start()
            procs.append(p)

        for i in range(len(arglist)):
            q.put(i)

        for p in range(num_workers):
            q.put(None)
        q.join()
        resultdict = {}
        for i in range(len(arglist)):
            res = retq.get()
            if not res == None:
                resultdict.update(res)
            else:
                exc_info = to_child.recv()
                print exc_info
                raise exc_info[1]
        for p in procs:
            p.join()
        OnOffSwitch['state'] = False
        return resultdict
    else:
        resultdict = {}
        for i in range(len(arglist)):
            resultdict.update({i:func(*(arglist[i]))})
        return resultdict
