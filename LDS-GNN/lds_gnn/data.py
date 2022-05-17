"""This module contains methods to load and manage datasets. For graph based data, it mostly resorts to gcn package"""

import numpy as np
import pandas as pd
from gcn.utils import load_data

from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

import scipy.sparse as sp

try:
    from lds_gnn.utils import Config, upper_triangular_mask
except ImportError as e:
    from utils import Config, upper_triangular_mask

import os
import sys
sys.path.append('../../')

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def permuteMat(aff, semantic): #permute affinity mat and semantic mat
    affInd = aff.index.tolist()
    semanticInd = semantic.index.tolist() #instead of index.values.tolist()
    Pind = [i for i in semanticInd if i in affInd]
    Qind = [i for i in affInd if i not in Pind]

    PMat = aff.loc[Pind].copy()
    QMat = aff.loc[Qind].copy()

    aff = pd.concat([PMat, QMat], axis=0)
    semanticInter = semantic.loc[Pind].copy()
    semanticOuter = semantic.drop(labels=Pind, axis=0)

    return aff, semanticInter, semanticOuter, Pind, Qind

def read_folds(dataset_str, fold):
    '''
    Reads previously saved train and validation indices.
    First line contains train idx, second line contains train idx.
    '''
    
    save_folder = '../../saved_folds/'
    
    with open(os.path.join(save_folder, f'{dataset_str}_{fold}')) as f:
        train_idx = f.readline().split(' ')
        val_idx = f.readline().split(' ')
        train_idx = list(map(int, train_idx))
        val_idx = list(map(int, val_idx))

    return train_idx, val_idx


class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        self.f1 = 'load_data_del_edges'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        if self.f2:
            res = eval(self.f2)(res, **self.kwargs_f2, seed=self.seed)
        return res


class EdgeDelConfigData(ConfigData):
    def __init__(self, **kwargs):
        self.prob_del = 0.5
        self.enforce_connected = True
        super().__init__(**kwargs)
        self.kwargs_f1['prob_del'] = self.prob_del
        if not self.enforce_connected:
            self.kwargs_f1['enforce_connected'] = self.enforce_connected
        del self.prob_del
        del self.enforce_connected


class UCI(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        self.n_es = None
        self.scale = None
        super().__init__(**kwargs)

    def load(self):
        if self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()

        if self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if self.scale:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
        ys = LabelBinarizer().fit_transform(y)

        if ys.shape[1] == 1:
            ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]

        from sklearn.model_selection import train_test_split

        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val + self.n_es,
                                                        test_size=n - self.n_train - self.n_val - self.n_es,
                                                        stratify=y)
        train, es, y_train, y_es = train_test_split(train, y_train, random_state=self.seed,
                                                    train_size=self.n_train + self.n_val, test_size=self.n_es,
                                                    stratify=y_train)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                      train_size=self.n_train, test_size=self.n_val,
                                                      stratify=y_train)

        train_mask = np.zeros([n, ], dtype=bool)
        train_mask[train] = True
        val_mask = np.zeros([n, ], dtype=bool)
        val_mask[val] = True
        es_mask = np.zeros([n, ], dtype=bool)
        es_mask[es] = True
        test_mask = np.zeros([n, ], dtype=bool)
        test_mask[test] = True

        return np.zeros([n, n]), np.zeros([n, n]), features, ys, train_mask, val_mask, es_mask, test_mask, None #None for class labels

class ImputationDataset(ConfigData):

    def __init__(self, **kwargs):
        self.n_train = None
        self.n_val = None
        self.n_es = None
        self.scale = None
        super().__init__(**kwargs)

    def load(self):

        dataset_str = self.dataset_name
        if dataset_str.startswith('fin'):
            _, dataset_size, embed_name = dataset_str.split('_')
            
            aff = pd.read_csv('../../dataset/finance/fin_{}/priceMat_{}.csv'.format(dataset_size, dataset_size), index_col = 0)
            if dataset_size == 'small':
                glove = pd.read_csv('../../dataset/finance/fin_{}/{}Mat.csv'.format(dataset_size, embed_name), sep = ',', index_col = 0).iloc[:,:-2]
            elif dataset_size == 'large':
                glove = pd.read_csv('../../dataset/finance/fin_{}/{}.txt'.format(dataset_size, embed_name), sep = ' ', index_col = 0, header = None)
        
        #applestore
        else:
            _, embed_name = dataset_str.split('_')    
            aff = pd.read_csv('../../dataset/applestore/affMat.csv', index_col = 0)
            glove = pd.read_csv('../../dataset/applestore/{}Mat.csv'.format(embed_name), sep = ',', index_col = 0).iloc[:,:-1]

        nfeats = aff.shape[1] - 1 #number of trading days, last column is the sector label
        aff, semanticInter, semanticOuter, Pind, Qind = permuteMat(aff, glove)
        class_labels = aff['y'].values #save to merge with feature matrix at the end
        aff.drop('y', axis = 1, inplace = True)
        glove_price = aff.join(semanticInter, how='left', lsuffix='_left', rsuffix='_right')
        glove_price[np.isnan(glove_price)] = 0 #fill missing embeds with 0 to be compatible.

        features = glove_price.iloc[:, :nfeats].values
        labels = glove_price.iloc[:, nfeats:].values
        nclasses = labels.shape[1]
        
        p, num_rows, train_ratio = len(semanticInter), labels.shape[0], 0.0
        idx_train, idx_val = read_folds(dataset_str = dataset_str, fold=0)
        idx_test = range(p, num_rows)
            
        train_mask = sample_mask(idx_train, num_rows)
        test_mask = sample_mask(idx_test, num_rows)
        val_mask = sample_mask(idx_val, num_rows)
        es_mask = sample_mask(idx_val, num_rows) #make it same with val_mask for now
        
        n = features.shape[0]
        
        return np.zeros([n,n]), np.zeros([n, n]), features, labels, train_mask, val_mask, es_mask, test_mask, class_labels

def reorganize_data_for_es(loaded_data, seed=0, es_n_data_prop=0.5):
    adj, adj_mods, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = loaded_data
    ys = y_train + y_val + y_test
    features = preprocess_features(features)
    msk1, msk2 = divide_mask(es_n_data_prop, np.sum(val_mask), seed=seed)
    mask_val = np.array(val_mask)
    mask_es = np.array(val_mask)
    mask_val[mask_val] = msk2
    mask_es[mask_es] = msk1

    return adj, adj_mods, features, ys, train_mask, mask_val, mask_es, test_mask


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def divide_mask(n1, n_tot, seed=0):
    rnd = np.random.RandomState(seed)
    p = n1 / n_tot if isinstance(n1, int) else n1
    chs = rnd.choice([True, False], size=n_tot, p=[p, 1. - p])
    return chs, ~chs
