import warnings
import torch
from citation_networks import sample_mask, load_finance_data, load_applestore_data

warnings.simplefilter("ignore")


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
