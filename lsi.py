'''
Latent Semantic Imputation (S. Yao et.al, SIGKDD 2019)
'''
import numpy as np
from hpc import multicore_dis, MSTKNN, multicore_nnls
import multiprocessing as mp


def iterSolveQ(P, W, ita=1e-4, verbose=False):

    mean = P.mean(axis = 0).reshape(1,-1) #mean along with row direction
    std = P.std(axis = 0).reshape(1,-1) #var along same direction
    N = W.shape[0]
    p = P.shape[0]
    q = N - p
    d = P.shape[1]
    W = W[-q:]
    Q = np.zeros((q,d))

    for i in range(q):
        noise = np.array([np.random.normal(0,s) for s in std])
        Q[i] = Q[i] + mean + noise

    PQ = np.vstack([P,Q])
    err = np.inf
    step = 0
    while err > ita: 
        dump = PQ[p:N].copy()
        PQ[p:N] = W.dot(PQ)
        err = sum(sum(abs(PQ[p:N]-dump))) / sum(sum(abs(dump)))
        if (verbose == True):
            print("%d %f" %(step, err))
            step += 1

    return PQ


def LSI(aff, semantic, delta, ita, spanning=True, verbose=False, n_jobs=-1):
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    N = aff.shape[0]
    P = semantic.shape[0]
    Q = N-P
    Q_index = range(N)#can change this to change the minimum spanning tree

    dis = multicore_dis(aff, Q_index, n_jobs)
    graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
    W = multicore_nnls(aff, graph, Q_index, n_jobs, epsilon=1e-1)
    PQ = iterSolveQ(semantic, W, ita, verbose)

    return PQ


if __name__ == '__main__':
    aff = np.random.rand(1000, 300)
    semantic = np.random.rand(500, 200)
    PQ = LSI(aff, semantic, 10, 1e-4, spanning=True, n_jobs=-1)
    print(PQ.shape)
