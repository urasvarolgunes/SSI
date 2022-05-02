# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import dgl
import torch.nn as nn

from layers import Diag
from utils import *
import time
import numpy as np


class MLP_Diag(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, non_linearity, sparse, mlp_act, num_anchor, device = 'cuda'):
        super(MLP_Diag, self).__init__()
        self.num_anchor = num_anchor
        self.graph_runtimes = []
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Diag(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.sparse = sparse
        self.mlp_act = mlp_act
        self.device = device
        
        if self.num_anchor > 0:
            print('using anchor KNN with m={}'.format(self.num_anchor))
            
    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            start_time = time.time()
            
            if self.num_anchor > 0:
                rows, cols, values = knn_fast_anchor(embeddings, self.k, 1000, m=self.num_anchor, device=self.device)
            else:
                rows, cols, values = knn_fast(embeddings, self.k, 1000, device=self.device)
            
            end_time = time.time()
            '''
            if len(self.graph_runtimes) < 5:      
                self.graph_runtimes.append(end_time-start_time)
                if len(self.graph_runtimes) == 5:
                    print('average graph time: {:6f}'.format(np.mean(self.graph_runtimes)))
                    print('std of graph time: {:6f}'.format(np.std(self.graph_runtimes, ddof=1)))
            '''       
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device=self.device)
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity)
            return similarities


class MLP(nn.Module):
    def __init__(self, nlayers, isize, hsize, osize, mlp_epochs, k, knn_metric, non_linearity, sparse, mlp_act):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, hsize))
        else:
            self.layers.append(nn.Linear(isize, hsize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(hsize, hsize))
            self.layers.append(nn.Linear(hsize, osize))

        self.input_dim = isize
        self.output_dim = osize
        self.mlp_epochs = mlp_epochs
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.mlp_knn_init()
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def mlp_knn_init(self):
        if self.input_dim == self.output_dim:
            print("MLP full")
            for layer in self.layers:
                layer.weight = nn.Parameter(torch.eye(self.input_dim))
        else:
            optimizer = torch.optim.Adam(self.parameters(), 0.01)
            labels = torch.from_numpy(nearest_neighbors(self.features.cpu(), self.k, self.knn_metric)).cuda()

            for epoch in range(1, self.mlp_epochs):
                self.train()
                logits = self.forward()
                loss = F.mse_loss(logits, labels, reduction='sum')
                if epoch % 10 == 0:
                    print("MLP loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity)
            return similarities
