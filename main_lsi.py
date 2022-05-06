import numpy as np
import sys
from hpc import *
from lsi import iterSolveQ
from data_loaders import load_data
import argparse
from utils import KNN
import os

parser = argparse.ArgumentParser()
parser.add_argument('-delta', type=int, default=20)
parser.add_argument('-dataset', type=str, default='fin_small_google')
parser.add_argument('-train_use_ratio', type=float, default=0)

args = parser.parse_args()

def test(X, y):
    N = [2,5,8,10,15,20,30]
    result = []
    for n in N: # classification n_neighbors = 5, n_components = 30 up
        result.append(KNN(X.copy(), y.copy(), n))
    return result


if __name__ == '__main__':

    all_results = []
    for seed in range(5):
        np.random.seed(seed)
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (X, y, p) = load_data(args)
        idx_train = list(range(p))
        labels = labels.numpy()
        
        graph_path = './saved_graphs/{}_{}.npy'.format(args.dataset, args.delta)
        if not os.path.exists(graph_path):
            Q_index = range(features.shape[0])
            dis = multicore_dis(features, Q_index, n_jobs=-1)
            graph = MSTKNN(dis, Q_index, args.delta, n_jobs=-1, spanning=True)
            adj = multicore_nnls(features, graph, Q_index, n_jobs=-1, epsilon=1e-1)
            adj = normalize_sp(adj + sp.eye(adj.shape[0]))
            np.save(graph_path, adj)
        else:
            adj = np.load(graph_path, allow_pickle=True).item()

        PQ = iterSolveQ(labels[idx_train], adj, 1e-3)
        result = test(PQ, y)
        result = [100*res for res in result]
        all_results.append(result)
    
    print(np.mean(all_results, axis=0))
    print(np.std(all_results, axis=0, ddof=1))