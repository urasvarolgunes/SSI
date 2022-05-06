import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data_loaders import load_data
from model import GCN
from utils import accuracy, get_random_mask, nearest_neighbors, normalize, KNN
import pdb
import time
EOS = 1e-10
import optuna
import math
from hpc import *
import os
import dgl

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='applestore_google', help='See choices',
                    choices=['applestore_google', 'applestore_glove',
                            'fin_small_google', 'fin_small_glove','fin_small_fast',
                            'fin_large_google', 'fin_large_glove','fin_large_fast'])
parser.add_argument('-device', type=str, default='cuda')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-sparse', type=int, default=1)
parser.add_argument('-ntrials', type=int, default=100)

parser.add_argument('-nlayers', type=int, default=2, help='#layers')
parser.add_argument('-epochs', type=int, default=1000, help='Total number of epochs')
parser.add_argument('-lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('-w_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('-hidden', type=int, default=600, help='Number of hidden units for GCN_R.')
parser.add_argument('-dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('-dropout_adj', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('-delta', type=int, default=20, help='number of neighbors for graph construction')
parser.add_argument('-train_use_ratio', type=float, default=0.0, help='percentage of training data to use')

args = parser.parse_args()


def train_end_to_end(fold, args, params, save_model, testing=False):
    
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (X, y, p) = load_data(args)
    
    graph_path = './saved_graphs/{}_{}_{}_{}.npy'.format(args.dataset, args.delta, args.seed, fold)
    if not os.path.exists(graph_path):
        Q_index = range(features.shape[0])
        dis = multicore_dis(features, Q_index, n_jobs=-1)
        graph = MSTKNN(dis, Q_index, args.delta, n_jobs=-1, spanning=True)
        adj = multicore_nnls(features, graph, Q_index, n_jobs=-1, epsilon=1e-1)
        adj = normalize_sp(adj + sp.eye(adj.shape[0]))
        np.save(graph_path, adj)
    else:
        adj = np.load(graph_path, allow_pickle=True).item()
    
    adj = adj.tocoo()
    Adj = dgl.graph((torch.LongTensor(adj.row), torch.LongTensor(adj.col)), num_nodes=len(features))
    Adj.edata['w'] = torch.FloatTensor(adj.data) #edge weights
    Adj = Adj.to(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    model = GCN(in_channels=nfeats,
                hidden_channels=args.hidden,
                out_channels=nclasses,
                num_layers=args.nlayers,
                dropout=params['dropout'],
                dropout_adj=params['dropout_adj'],
                sparse = args.sparse,
                Adj = Adj,
                )

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])

    if torch.cuda.is_available() and args.device =='cuda':
        model = model.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()
        val_mask = val_mask.cuda()
        features = features.cuda()
        labels = labels.cuda()
        if testing:
            train_mask = train_mask + val_mask #in final training, we use validation set too
    
    best_loss = math.inf
    early_stop_tolerance = 20
    early_stop_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(features)
        loss = F.mse_loss(preds[train_mask], labels[train_mask], reduction='mean')
        loss.backward()
        optimizer.step()

        print("Epoch {:05d} | Training Loss: {:.4f}".format(epoch, loss.item()))
        
        #early stop
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(preds[val_mask], labels[val_mask], reduction='mean').item()
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_count = 0
                if save_model:
                    torch.save(model.state_dict(), f"saved_models/best_model_{fold}.bin")
            else:
                early_stop_count += 1
            
            if early_stop_count == early_stop_tolerance:
                print('early stopping...')
                break
        

    model.load_state_dict(torch.load(f'saved_models/best_model_{fold}.bin'))

    with torch.no_grad():
        model.eval()
        preds = model(features)
        val_loss = F.mse_loss(preds[val_mask], labels[val_mask], reduction='mean').item()
        test_preds = preds.detach().cpu().numpy()[test_mask.cpu()]

    return val_loss, test_preds

def place_embeddings(X, y, preds, p):
    X[p:,:] = preds #replace unknown embeddings (first p is known) with predictions
    print('Imptutation is finished. Performing KNN evaluation...')
    knn_dict = eval_knn(X, y, k_list = [5,10,20])
    print(knn_dict)


def eval_knn(X, y, k_list):
    knn_dict = dict()
    for k in k_list:
        knn_acc = KNN(X, y, n=k)
        knn_dict[k] = round(knn_acc*100, 2)

    return knn_dict


def objective(trial):
    params = {"dropout": trial.suggest_float(name="dropout", low=0.0, high=0.5, step=0.1),
                "dropout_adj": trial.suggest_float(name="dropout_adj", low=0.0, high=0.5, step=0.1),              
                #"k": trial.suggest_int(name="k", low=10, high=30, step=10),
                "lr": trial.suggest_loguniform(name='lr', low=1e-5, high=1e-3),
                "w_decay": trial.suggest_loguniform(name='w_decay', low=1e-7, high=1e-4),
                #"epochs": trial.suggest_int(name='epochs', low=50, high=500, step=25),
            }
    
    loss, test_preds = train_end_to_end(fold=0, args=args, params=params, save_model=True)

    return loss
         
if __name__ == '__main__':    
    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.ntrials)
    
    print("best_trial:")
    best_trial = study.best_trial
    
    print(best_trial.values)
    print(best_trial.params)
    
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (X, y, p) = load_data(args)
    test_loss, test_preds = train_end_to_end(fold=0, args=args, params=best_trial.params, save_model=True, testing=False)
    place_embeddings(X, y, test_preds, p)

    end_time = time.time()
    print('one run takes {} seconds.'.format(end_time-start_time))