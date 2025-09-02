import networkx as nx
import multiprocessing as mp
import random
import sys
import itertools as IT
import logging
logger = mp.log_to_stderr(logging.DEBUG)


def worker(inqueue, output):
    result = []
    count = 0
    for pair in iter(inqueue.get, sentinel):
        source, target = pair
        for path in nx.all_simple_paths(G, source = source, target = target,
                                        cutoff = None):
            result.append(path)
            count += 1
            if count % 10 == 0:
                logger.info('{c}'.format(c = count))
    output.put(result)

def test_workers():
    result = []
    inqueue = mp.Queue()
    for source, target in IT.product(sources, targets):
        inqueue.put((source, target))
    procs = [mp.Process(target = worker, args = (inqueue, output))
             for i in range(mp.cpu_count())]
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        inqueue.put(sentinel)
    for proc in procs:
        result.extend(output.get())
    for proc in procs:
        proc.join()
    return result

def test_single_worker():
    result = []
    count = 0
    for source, target in IT.product(sources, targets):
        for path in nx.all_simple_paths(G, source = source, target = target,
                                        cutoff = None):
            result.append(path)
            count += 1
            if count % 10 == 0:
                logger.info('{c}'.format(c = count))

    return result

sentinel = None

seed = 1
m = 1
N = 1340//m
G = nx.gnm_random_graph(N, int(1.7*N), seed)
random.seed(seed)
sources = [random.randrange(N) for i in range(340//m)]
targets = [random.randrange(N) for i in range(1000//m)]
output = mp.Queue()

if __name__ == '__main__':
    test_workers()
    # test_single_worker()
    # assert set(map(tuple, test_workers())) == set(map(tuple, test_single_worker()))