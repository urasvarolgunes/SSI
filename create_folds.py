import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import permuteMat
from sklearn.utils.multiclass import type_of_target
import numpy as np

def split_fin_small(embedding_name, dataset_size):
    
    aff = pd.read_csv(f'dataset/finance/fin_{dataset_size}/priceMat_{dataset_size}.csv', index_col=0)

    if dataset_size == 'small':
        glove = pd.read_csv(f'dataset/finance/fin_{dataset_size}/{embedding_name}Mat.csv', sep = ',', index_col=0).iloc[:,:-2]
    elif dataset_size == 'large':
        glove = pd.read_csv(f'dataset/finance/fin_{dataset_size}/{embedding_name}.txt', sep = ' ', index_col = 0, header = None)

    aff, semanticInter, semanticOuter, Pind, Qind = permuteMat(aff, glove)
    targets = aff['y'].values
    aff = aff.drop('y', axis=1)
    last_train_index = len(semanticInter)
    kf = StratifiedKFold(n_splits=5)
    for fold_num, (train_idx,val_idx) in enumerate(kf.split(X=aff[:last_train_index], y=targets[:last_train_index])):
        with open('saved_folds/fin_{}_{}_{}'.format(dataset_size, embedding_name, fold_num), 'w') as f:
            f.write(' '.join(map(str, train_idx.tolist())))
            f.write('\n')
            f.write(' '.join(map(str, val_idx.tolist())))

def split_applestore(embedding_name):

    aff = pd.read_csv('dataset/applestore/affMat.csv', index_col = 0)
    glove = pd.read_csv('dataset/applestore/{}Mat.csv'.format(embedding_name), sep = ',', index_col = 0).iloc[:,:-1]
    aff, semanticInter, semanticOuter, Pind, Qind = permuteMat(aff, glove)
    targets = aff['y'].values
    aff = aff.drop('y', axis=1)
    last_train_index = len(semanticInter)
    kf = StratifiedKFold(n_splits=5)
    for fold_num, (train_idx,val_idx) in enumerate(kf.split(X=aff[:last_train_index], y=targets[:last_train_index])):
        print([np.unique(targets[train_idx], return_counts=True)], np.unique(targets[val_idx], return_counts=True))
        with open('saved_folds/applestore_{}_{}'.format(embedding_name, fold_num), 'w') as f:
            f.write(' '.join(map(str, train_idx.tolist())))
            f.write('\n')
            f.write(' '.join(map(str, val_idx.tolist())))

def read_folds(fold, dataset):
    '''
    Reads previously saved train and validation indices.
    First line contains train idx, second line contains train idx.
    '''
    pass

if __name__ == "__main__":

    for embedding_name in ['google','glove','fast']:
        for dataset_size in ['small','large']:
            split_fin_small(embedding_name, dataset_size)

    #fast contains insufficient number of training samples
    for embedding_name in ['google','glove']:    
        split_applestore(embedding_name)
    
