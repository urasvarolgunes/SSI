# Embedding-Imputation-with-Self-Supervised-GNNs
Official repository for "Embedding Imputation with Self-Supervised GNNs". <br />
Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10173511

## Datasets

- Industry Category Classification: fin_large_google, fin_large_glove, fin_large_fast, fin_small_google, fin_small_glove, fin_small_fast
- The Mobile App Statistics dataset: applestore_google, applestore_glove

## Dependencies

* `Python` version 3.7.2
* [`Numpy`](https://numpy.org/) version 1.18.5
* [`PyTorch`](https://pytorch.org/) version 1.5.1
* [`DGL`](https://www.dgl.ai/) version 0.5.2
* [`sklearn`](https://scikit-learn.org/stable/) version 0.21.3
* [`scipy`](https://www.scipy.org/) version 1.2.1
* [`torch-geometric`](https://github.com/rusty1s/pytorch_geometric) 1.6.1

CUDA version is 11.3
You can create a virtual environment and install all the dependencies with the following command:

```bash
conda env create -f environment.yml
```
## Usage

To run the model you should define the following parameters:

- `dataset`: The dataset name
- `device`: cpu or cuda
- `ntrials`: number of runs
- `epochs_adj`: number of total epochs
- `lr_adj`: learning rate of GCN_S
- `lr`: learning rate of GCN_R
- `w_decay_adj`: l2 regularization parameter for GCN_S
- `w_decay`: l2 regularization parameter for GCN_R
- `nlayers_adj`: number of layers for GCN_S
- `nlayers`: number of layers for GCN_R
- `hidden_adj`: hidden size of GCN_S
- `hidden`: hidden size of GCN_R
- `dropout1`: dropout rate for GCN_S
- `dropout2`: dropout rate for GCN_R
- `dropout_adj1`: dropout rate on adjacency matrix for GCN_S
- `dropout_adj2`: dropout rate on adjacency matrix for GCN_R
- `k`: k for building the KNN-graph
- `num_anchor`: number of anchors for building the Anchor-KNN-graph, set to 0 to disable
- `lambda_`: weight of the self-supervision loss
- `ratio`: ratio of features to mask out
- `sparse`: whether to make the adjacency sparse and run operations on sparse mode
- `non_linearity`: non-linearity to apply on the adjacency matrix
- `mlp_act`: activation function to use for the mlp graph generator
- `mlp_h`: hidden size of the mlp graph generator
- `noise`: type of noise to add to features (mask or normal)
- `epoch_d`: First (epochs_adj / epoch_d) of the epochs will be used only for training GCN_S

## Reproducing the Results

In order to reproduce the results presented in the paper, you should run the following commands. To use Anchor-KNN, add the `-num_anchor` flag and specify the number of anchors.

### Industry Category Classification (Large)

#### word2vec (fin_large_google)

```bash
python main_finance.py -dataset=fin_large_google -device=cuda -seed=1 -epochs_adj=125 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=0 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-4 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse=1 -gen_mode=2 -non_linearity=relu -mlp_act=tanh -epoch_d=5
```

#### GloVe (fin_large_glove)

```bash
python main_finance.py -dataset=fin_large_glove -device=cuda -seed=1 -epochs_adj=175 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=0 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-4 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse=1 -gen_mode=2 -non_linearity=relu -mlp_act=tanh -epoch_d=5 -noise=mask
```

#### fastText (fin_large_fast)

```bash
python main_finance.py -dataset=fin_large_fast -device=cuda -seed=1 -epochs_adj=150 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=0 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-4 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse=1 -gen_mode=2 -non_linearity=relu -mlp_act=tanh -epoch_d=5 -noise=mask
```

### Industry Category Classification (Small)

#### word2vec (fin_small_google)

```bash
python main_finance.py -dataset=fin_small_google -epochs_adj=350 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=0 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-3 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse=1 -non_linearity=relu -mlp_act=tanh -epoch_d=5
```

#### GloVe (fin_small_glove)

```bash
python main_finance.py -dataset=fin_small_google -epochs_adj=300 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=1e-6 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-3 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse=1 -non_linearity=relu -mlp_act=tanh -epoch_d=5
```

#### fastText (fin_small_fast)

```bash
python main_finance.py -dataset=fin_small_google -epochs_adj=350 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=1e-6 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-4 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse=1 -non_linearity=relu -mlp_act=tanh -epoch_d=5
```

### Mobile Statistics Dataset

#### word2vec (applestore_google)

```bash
python main_applestore.py -dataset=applestore_google -epochs_adj=400 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=0 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-3 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse 1 -non_linearity=relu -mlp_act=tanh -epoch_d=5
```

#### GloVe (applestore_glove)

```bash
python main_applestore.py -dataset=applestore_glove -epochs_adj=400 -dropout1=0 -dropout2=0 -dropout_adj1=0 -dropout_adj2=0 -w_decay=0 -lambda_=1 -ratio=5 -lr=1e-3 -lr_adj=1e-3 -nlayers=2 -nlayers_adj=2 -hidden=600 -hidden_adj=600 -mlp_h=300 -k=20 -sparse 1 -non_linearity=relu -mlp_act=tanh -epoch_d=5
```
## Citation
If you find the work useful please consider citing us.

```
@ARTICLE{10173511,
  author={Varolgunes, Uras and Yao, Shibo and Ma, Yao and Yu, Dantong},
  journal={IEEE Access}, 
  title={Embedding Imputation With Self-Supervised Graph Neural Networks}, 
  year={2023},
  volume={11},
  number={},
  pages={70610-70620},
  keywords={Embedded systems;Graph neural networks;Natural language processing;Embedded systems;Self-supervised learning;Embedding imputation;graph neural networks;natural language processing},
  doi={10.1109/ACCESS.2023.3292314}}
```
