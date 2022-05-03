import pickle as pkl
import sys
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import pdb
warnings.simplefilter("ignore")

from utils import permuteMat
import os
from create_folds import read_folds

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_applestore_data(dataset_str, train_use_ratio=0, fold=0):

    _, embed_name = dataset_str.split('_')
    
    aff = pd.read_csv('dataset/applestore/affMat.csv', index_col = 0)
    glove = pd.read_csv('dataset/applestore/{}Mat.csv'.format(embed_name), sep = ',', index_col = 0).iloc[:,:-1]

    nfeats = aff.shape[1] - 1 #number of trading days, last column is the sector label

    aff, semanticInter, semanticOuter, Pind, Qind = permuteMat(aff, glove)
    class_labels = aff['y'].values #save to merge with feature matrix at the end
    aff.drop('y', axis = 1, inplace = True)
    glove_price = aff.join(semanticInter, how='left', lsuffix='_left', rsuffix='_right')
    glove_price[np.isnan(glove_price)] = 0 #fill missing embeds with 0 to be compatible.

    features = glove_price.iloc[:, :nfeats]
    labels = glove_price.iloc[:, nfeats:]
    labels_numpy = labels.copy().values
    nclasses = labels.shape[1]
    
    p, num_rows, train_ratio = len(semanticInter), labels.shape[0], 0.0
    idx_train, idx_val = read_folds(dataset='applestore', embedding_name=embed_name, fold=fold)
    idx_test = range(p, num_rows)
        
    if train_use_ratio != 0:
        thres_row = int(train_use_ratio*num_rows)
        assert thres_row < p, "Do not have that much training data, reduce train_use_ratio!"
        p = thres_row # p is later used for KNN evaluation, need to change it when trimming some training data
        idx_train = range(0, thres_row)
        idx_test = range(thres_row, num_rows)
        
    train_mask = sample_mask(idx_train, num_rows)
    test_mask = sample_mask(idx_test, num_rows)
    val_mask = sample_mask(idx_val, num_rows)
    
    features = torch.FloatTensor(features.values)
    labels = torch.FloatTensor(labels.values)
    train_mask = torch.BoolTensor(train_mask)    
    test_mask = torch.BoolTensor(test_mask)
    val_mask = torch.BoolTensor(val_mask)
    
    '''
    print('train: {}, val: {}, test: {}'.format(train_mask.sum(), 0, test_mask.sum()))
    '''
    
    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (labels_numpy, class_labels, p)
    
def load_finance_data(dataset_str, train_use_ratio=0, use_validation = False, fold=0):

    _, dataset_size, embed_name = dataset_str.split('_')
    
    aff = pd.read_csv('dataset/finance/fin_{}/priceMat_{}.csv'.format(dataset_size, dataset_size), index_col = 0)
    if dataset_size == 'small':
        glove = pd.read_csv('dataset/finance/fin_{}/{}Mat.csv'.format(dataset_size, embed_name), sep = ',', index_col = 0).iloc[:,:-2]
    elif dataset_size == 'large':
        glove = pd.read_csv('dataset/finance/fin_{}/{}.txt'.format(dataset_size, embed_name), sep = ' ', index_col = 0, header = None)

    nfeats = aff.shape[1] - 1 #number of trading days, last column is the sector label

    aff, semanticInter, semanticOuter, Pind, Qind = permuteMat(aff, glove)
    class_labels = aff['y'].values #save to merge with feature matrix at the end
    aff.drop('y', axis = 1, inplace = True)
    glove_price = aff.join(semanticInter, how='left')
    glove_price[np.isnan(glove_price)] = 0 #fill missing embeds with 0 to be compatible.

    features = glove_price.iloc[:, :nfeats]
    labels = glove_price.iloc[:, nfeats:]
    labels_numpy = labels.copy().values
    nclasses = labels.shape[1]
    
    p, num_rows, train_ratio = len(semanticInter), labels.shape[0], 0.95
    #idx_train = range(0,p)
    idx_train, idx_val = read_folds(fold=fold, dataset=f'fin_{dataset_size}', embedding_name=embed_name)
    idx_test = range(p, num_rows)
    val_mask = None
    
    if train_use_ratio != 0:
        thres_row = int(train_use_ratio*num_rows)
        assert thres_row < p, "Do not have that much training data, reduce train_use_ratio!"
        p = thres_row # p is later used for KNN evaluation, need to change it when trimming some training data
        idx_train = range(0, thres_row)
        idx_test = range(thres_row, num_rows)
        val_mask = None
    
    train_mask = sample_mask(idx_train, num_rows)
    test_mask = sample_mask(idx_test, num_rows)
    
    features = torch.FloatTensor(features.values)
    labels = torch.FloatTensor(labels.values)
    train_mask = torch.BoolTensor(train_mask)    
    test_mask = torch.BoolTensor(test_mask)
    
    '''
    print('train: {}, val: {}, test: {}'.format(train_mask.sum(), 0, test_mask.sum()))
    '''
    
    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, (labels_numpy, class_labels, p)


def load_data(args):
    finance_datasets = ['fin_small_glove','fin_small_google','fin_small_fast','fin_large_glove','fin_large_google','fin_large_fast']
    applestore_datasets = ['applestore_google', 'applestore_glove']
    dataset_str = args.dataset
    train_use_ratio = args.train_use_ratio
    
    if dataset_str in finance_datasets:
        return load_finance_data(dataset_str, train_use_ratio)
        
    elif dataset_str in applestore_datasets:
        return load_applestore_data(dataset_str, train_use_ratio)    
    
    else:
        raise NameError("Wrong dataset name")