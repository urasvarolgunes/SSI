# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data_loaders import load_data
from model import GCN, GCN_R, GCN_S
from utils import accuracy, get_random_mask, nearest_neighbors, normalize, KNN
import pdb
import time
EOS = 1e-10
import optuna
import math

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='applestore_google', help='See choices',
                    choices=['applestore_google', 'applestore_glove',
                            'fin_small_google', 'fin_small_glove','fin_small_fast',
                            'fin_large_google', 'fin_large_glove','fin_large_fast'])
parser.add_argument('-device', type=str, default='cuda')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-sparse', type=int, default=1)
parser.add_argument('-ntrials', type=int, default=100)

parser.add_argument('-knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
parser.add_argument('-nlayers', type=int, default=2, help='#layers')
parser.add_argument('-nlayers_adj', type=int, default=2, help='#layers')
parser.add_argument('-gen_mode', type=int, default=2)
parser.add_argument('-normalization', type=str, default='sym')
parser.add_argument('-mlp_h', type=int, default=300)

parser.add_argument('-epochs_adj', type=int, default=500, help='Total number of epochs')
parser.add_argument('-lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('-lr_adj', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('-w_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('-w_decay_adj', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('-hidden', type=int, default=600, help='Number of hidden units for GCN_R.')
parser.add_argument('-hidden_adj', type=int, default=600, help='Number of hidden units for GCN_S.')
parser.add_argument('-dropout1', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('-dropout2', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('-dropout_adj1', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('-dropout_adj2', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('-k', type=int, default=20, help='k for initializing with knn')
parser.add_argument('-ratio', type=int, default=5, help='ratio of ones to select for each mask')
parser.add_argument('-epoch_d', type=float, default=5, help='epochs_adj / epoch_d of the epochs will be used for training only with GCN_S.')
parser.add_argument('-lambda_', type=float, default=1, help='self-supervision strength')
parser.add_argument('-non_linearity', type=str, default='relu')
parser.add_argument('-mlp_act', type=str, default='tanh', choices=["relu", "tanh"])
parser.add_argument('-mlp_epochs', type=int, default=100)
parser.add_argument('-noise', type=str, default="mask", choices=['mask', 'normal'])

parser.add_argument('-num_anchor', type=int, help='set to 0 to stop using anchor knn', default=0)
parser.add_argument('-train_use_ratio', type=float, default=0.0, help='percentage of training data to use')
args = parser.parse_args()

def get_loss_learnable_adj(model, mask, features, labels, Adj):
    logits = model(features, Adj)        
    loss = F.mse_loss(logits[mask], labels[mask], reduction='mean')
    return loss, logits[mask] 

def get_loss_masked_features(model, features, mask, noise):

    if noise == 'mask':
        masked_features = features * (1 - mask)
    elif noise == "normal":
        noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
        masked_features = features + (noise * mask)

    logits, Adj = model(features, masked_features)

    indices = mask > 0
    loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        
    return loss, Adj


def train_end_to_end(fold, args, params, save_model, testing=False):
    
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (X, y, p) = load_data(args, fold=fold)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    model1 = GCN_S(sparse=args.sparse, #fixed
                    knn_metric=args.knn_metric, #fixed
                    features=features.cpu(), #fixed
                    nlayers=args.nlayers_adj, #fixed
                    in_dim=nfeats, #fixed,
                    mlp_epochs=args.mlp_epochs, #only for gen_mode=1
                    mlp_h=args.mlp_h, #only for gen_mode=1
                    normalization=args.normalization, #fixed, symmetric
                    nclasses=nfeats, #fixed
                    gen_mode=args.gen_mode, #fixed, 2
                    hidden_dim=args.hidden_adj, #fixed, hidden units for GCN
                    non_linearity= args.non_linearity, #fixed, relu
                    dropout=params['dropout1'],
                    dropout_adj=params['dropout_adj1'],
                    mlp_act=params['mlp_act'], #tanh or relu
                    k=params['k'], #for building knn graph
                    num_anchor=args.num_anchor, #only needed when using anchor graph
                    )

    model2 = GCN_R(sparse=args.sparse, #fixed
                   in_channels=nfeats, #fixed
                   out_channels=nclasses, #fixed
                   hidden_channels=args.hidden, #fixed
                   num_layers=args.nlayers, #fixed
                   dropout=params['dropout2'],
                   dropout_adj=params['dropout_adj2'],
                   )

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=params['lr_adj'], weight_decay=params['w_decay_adj'])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=params['lr'], weight_decay=params['w_decay'])

    if torch.cuda.is_available() and args.device =='cuda':
        model1 = model1.cuda()
        model2 = model2.cuda()
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

    for epoch in range(1, args.epochs_adj + 1):
        model1.train()
        model2.train()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        mask = get_random_mask(features, args.ratio).to(args.device)

        if epoch < args.epochs_adj // args.epoch_d:
            model2.eval()
            loss1, Adj = get_loss_masked_features(model1, features, mask, args.noise)
            loss2 = torch.tensor(0).cuda()
        else:
            loss1, Adj = get_loss_masked_features(model1, features, mask, args.noise)
            loss2, preds = get_loss_learnable_adj(model2, train_mask, features, labels, Adj)

        loss = loss1 * params['lambda_'] + loss2
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        
        if epoch > (args.epochs_adj // args.epoch_d):
            #print("Epoch {:05d} | GCN_S Loss: {:.4f}, GCN_R Loss: {:.4f}".format(epoch, loss1.item() * args.lambda_, loss2.item()))
            
            #early stop
            model2.eval()
            with torch.no_grad():
                val_loss, preds = get_loss_learnable_adj(model2, val_mask, features, labels, Adj)
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_count = 0
                    if save_model:
                        torch.save(model2.state_dict(), f"saved_models/best_model_{fold}.bin")
                else:
                    early_stop_count += 1
                
                if early_stop_count == early_stop_tolerance:
                    #print('early stopping...')
                    break
            
    model2.load_state_dict(torch.load(f'saved_models/best_model_{fold}.bin'))
    with torch.no_grad():
        model2.eval()
        val_loss, preds = get_loss_learnable_adj(model2, val_mask, features, labels, Adj)
        test_loss, test_preds = get_loss_learnable_adj(model2, test_mask, features, labels, Adj)
        test_preds = test_preds.detach().cpu().numpy()

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
    params = {"dropout1": trial.suggest_float(name="dropout1", low=0.0, high=0.5, step=0.1),
                "dropout2": trial.suggest_float(name="dropout2", low=0.0, high=0.5, step=0.1),
                "dropout_adj1": trial.suggest_float(name="dropout_adj1", low=0.0, high=0.5, step=0.1),
                "dropout_adj2": trial.suggest_float(name="dropout_adj2", low=0.0, high=0.5, step=0.1),
                "mlp_act": trial.suggest_categorical(name="mlp_act", choices=['tanh']),                
                "k": trial.suggest_int(name="k", low=10, high=30, step=10),
                "lr": trial.suggest_loguniform(name='lr', low=1e-5, high=1e-3),
                "lr_adj": trial.suggest_loguniform(name='lr_adj', low=1e-5, high=1e-3),
                "w_decay": trial.suggest_loguniform(name='w_decay', low=1e-7, high=1e-4),
                "w_decay_adj": trial.suggest_loguniform(name='w_decay_adj', low=1e-7, high=1e-4),
                #"epochs_adj": trial.suggest_int(name='epochs_adj', low=50, high=500, step=25),
                "lambda_": trial.suggest_int(name='lambda_', low=1, high=10, step=1),
            }
    
    losses = []
    for fold in [0]:
        loss, test_preds = train_end_to_end(fold=fold, args=args, params=params, save_model=True)
        losses.append(loss.item())
    
    return np.mean(losses)
         
if __name__ == '__main__':    
    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.ntrials)
    
    print("best_trial:")
    best_trial = study.best_trial
    
    print(best_trial.values)
    print(best_trial.params)
    
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (X, y, p) = load_data(args)
    test_loss, test_preds = train_end_to_end(fold=0, args=args, params=best_trial.params, save_model=True, testing=True)
    place_embeddings(X, y, test_preds, p)

    end_time = time.time()
    print('one run takes {} seconds.'.format(end_time-start_time))