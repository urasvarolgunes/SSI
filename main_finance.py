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

from data_loader import load_data
from model import GCN, GCN_R, GCN_S
from utils import accuracy, get_random_mask, nearest_neighbors, normalize, KNN
import pdb
import time
EOS = 1e-10


class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def get_loss_learnable_adj(self, model, mask, features, labels, Adj):
        logits = model(features, Adj)        
        loss = F.mse_loss(logits[mask], labels[mask], reduction='mean')
        return loss, logits[mask] 

    def get_loss_masked_features(self, model, features, mask, noise):

        if noise == 'mask':
            masked_features = features * (1 - mask)
        elif noise == "normal":
            noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
            masked_features = features + (noise * mask)

        logits, Adj = model(features, masked_features)

        indices = mask > 0
        loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
            
        return loss, Adj

    def train_end_to_end(self, args):
        
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (X, y, p) = load_data(args)
        
        for trial in range(args.ntrials):
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available() and args.device == 'cuda':
                torch.cuda.manual_seed(args.seed)
                
            model1 = GCN_S(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj, nclasses=nfeats,
                             dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                             features=features.cpu(), k=args.k, knn_metric=args.knn_metric,
                             non_linearity=args.non_linearity, normalization=args.normalization, mlp_h=args.mlp_h,
                             mlp_epochs=args.mlp_epochs, gen_mode=args.gen_mode, sparse=args.sparse,
                             mlp_act=args.mlp_act, num_anchor=args.num_anchor, device=args.device)
            model2 = GCN_R(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)

            optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available() and args.device =='cuda':

                model1 = model1.cuda()
                model2 = model2.cuda()
                train_mask = train_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

            for epoch in range(1, args.epochs_adj + 1):
                model1.train()
                model2.train()

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                
                mask = get_random_mask(features, args.ratio).to(args.device)

                if epoch < args.epochs_adj // args.epoch_d:
                    model2.eval()
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, args.noise)
                    loss2 = torch.tensor(0).to(args.device)
                else:
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, args.noise)
                    loss2, preds = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj)

                loss = loss1 * args.lambda_ + loss2
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                
                if epoch > args.epochs_adj // args.epoch_d and epoch % 20 == 0:
                    print("Epoch {:05d} | GCN_S Loss: {:.4f}, GCN_R Loss: {:.4f}".format(epoch, loss1.item() * args.lambda_, loss2.item()))
            
            #KNN evaluation
            model2.eval()
            with torch.no_grad():
                test_loss_, preds = self.get_loss_learnable_adj(model2, test_mask, features, labels, Adj)
                
            X[p:,(X.shape[1]-300):] = preds.cpu().numpy()
            print('Imptutation is finished. Performing KNN evaluation...')
            knn_dict = self.eval_knn(X, y, k_list = [2,5,8,10,15,20,30]) #[2,5,8,10,15,20,30]
            print(knn_dict)
            
    def eval_knn(self, X, y, k_list):   
        knn_dict = dict()
        for k in k_list:
            knn_acc = KNN(X, y, n=k)
            knn_dict[k] = round(knn_acc*100, 2)

        return knn_dict
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-epochs_adj', type=int, default=125, help='Number of epochs to learn the adjacency.')
    parser.add_argument('-lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('-lr_adj', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-w_decay_adj', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-hidden', type=int, default=600, help='Number of hidden units.')
    parser.add_argument('-hidden_adj', type=int, default=600, help='Number of hidden units.')
    parser.add_argument('-dropout1', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout2', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj1', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj2', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dataset', type=str, default='fin_large_google', help='See choices',
                    choices=['fin_small_glove','fin_small_google','fin_small_fast','fin_large_glove','fin_large_google','fin_large_fast'])
    parser.add_argument('-nlayers', type=int, default=2, help='#layers')
    parser.add_argument('-nlayers_adj', type=int, default=2, help='#layers')
    parser.add_argument('-ntrials', type=int, default=1, help='Number of trials')
    parser.add_argument('-k', type=int, default=20, help='k for initializing with knn')
    parser.add_argument('-ratio', type=int, default=5, help='ratio of ones to select for each mask')
    parser.add_argument('-epoch_d', type=float, default=5,
                        help='epochs_adj / epoch_d of the epochs will be used for training only with GCN_S.')
    parser.add_argument('-lambda_', type=float, default=1.0, help='ratio of ones to take')
    parser.add_argument('-knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
    parser.add_argument('-non_linearity', type=str, default='relu')
    parser.add_argument('-mlp_act', type=str, default='tanh', choices=["relu", "tanh"])
    parser.add_argument('-normalization', type=str, default='sym')
    parser.add_argument('-mlp_h', type=int, default=300)
    parser.add_argument('-mlp_epochs', type=int, default=100)
    parser.add_argument('-noise', type=str, default="mask", choices=['mask', 'normal'])
    parser.add_argument('-gen_mode', type=int, default=2, help='identifies the graph generator')
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-mask', type=str, default="mask", choices=['mask', 'normal'])
    parser.add_argument('-num_anchor', type=int, help='set to 0 to stop using anchor knn', default=0)
    parser.add_argument('-train_use_ratio', type=float, default=0.0, help='percent of training data to use, set to 0 to use all available training data.')
    args = parser.parse_args()
    
    experiment = Experiment()
    experiment.train_end_to_end(args)
