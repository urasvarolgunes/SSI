"""THe module contains the main algorithm and a demo script to run it"""

import math
import gcn.metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier

try:
    from lds_gnn.data import ConfigData, UCI, EdgeDelConfigData, ImputationDataset
    from lds_gnn.models import dense_gcn_2l_tensor
    from lds_gnn.utils import *
    from lds_gnn.hyperparams import *
except ImportError as e:
    # noinspection PyUnresolvedReferences
    from utils import *
    # noinspection PyUnresolvedReferences
    from hyperparams import *
    # noinspection PyUnresolvedReferences
    from data import ConfigData, UCI, EdgeDelConfigData, ImputationDataset
    # noinspection PyUnresolvedReferences
    from models import dense_gcn_2l_tensor


imputation_datasets = ['fin_small_google', 'fin_small_glove', 'fin_small_fast',
                        'fin_large_google', 'fin_large_glove', 'fin_large_fast',
                        'applestore_google','applestore_glove']

def from_svd(svd, train=False, ss=None):
    svd['config'].train = train
    _vrs = eval('Methods.' + svd['method'])(svd['data_config'], svd['config'])
    if not train: restore_from_svd(svd, ss)
    return _vrs


def empirical_mean_model(S, sample_vars, model_out, *what, fd=None, ss=None):
    """ Computes the tensors in `what` using the empirical mean output of the model given by
    `model_out`, sampling `S` times the stochastic variables in `sample_vars`"""
    if ss is None: ss = tf.get_default_session()
    smp = [sample(h) for h in far.utils.as_list(sample_vars)]
    mean_out = []
    for i in range(S):
        ss.run(smp)
        mean_out.append(ss.run(model_out, fd))
    mean_out = np.mean(mean_out, axis=0)
    lst = ss.run(what, {**fd, model_out: mean_out})
    return lst[0] if len(lst) == 1 else lst


class ConfigMethod(Config):
    def __init__(self, method_name=None, **kwargs):
        self.method_name = method_name
        self.seed = 1979
        self.train = True
        super().__init__(**kwargs)

    def execute(self, data_conf, **kwargs):
        return eval('Methods.' + self.method_name)(data_config=data_conf, config=self, **kwargs)


class LDSConfig(ConfigMethod):

    def __init__(self, method_name='lds', **kwargs):
        self.est_p_edge = 0.0  # initial estimation for the probability of an edge
        self.io_lr = 1e-4  # learning rate for the inner optimizer (either hyperparameter sigmoid
        # or fixed value if float
        self.io_opt = 'far.AdamOptimizer'  # name of the inner objective optimizer (should be a far.Optimizer)
        self.oo_lr = 1e-4  # learning rate of the outer optimizer # decreasing learning rate if this is a tuple
        self.oo_opt = 'far.GradientDescentOptimizer'  # name of the outer  objective optimizer
        self.io_steps = 5  # number of steps of the inner optimization dynamics before an update (hyper batch size)
        self.io_params = (0.001, 20, 400)  # minimum decrease coeff, patience, maxiters
        self.pat = 20  # patience for early stopping
        self.n_sample = 16  # number of samples to compute early stopping validation accuracy and test accuracy
        self.l2_reg = 1e-7  # l2 regularization coefficient (as in Kipf, 2017 paper)
        self.keep_prob = 1.  # also this is probably not really needed
        self.num_layer = 2

        super().__init__(method_name=method_name, _version=2, **kwargs)

    def io_optimizer(self) -> far.Optimizer:
        if isinstance(self.io_lr, tuple):
            c, a, b = self.io_lr  # starting value, minimum, maximum  -> re-parametrized as a + t * (b-a) with
            # t = sigmoid(\lambda)
            lr = a + tf.sigmoid(far.get_hyperparameter('lr', -tf.log((b - a) / (c - a) - 0.99))) * (b - a)
        else:
            lr = tf.identity(self.io_lr, name='lr')
        lr = tf.identity(lr, name='io_lr')
        return eval(self.io_opt)(lr)

    def oo_optimizer(self, multiplier=1.):
        opt_f = eval(self.oo_opt)
        if isinstance(self.oo_lr, float):
            lr = tf.identity(multiplier * self.oo_lr, name='o_lrd')
            return opt_f(lr)
        elif isinstance(self.oo_lr, tuple):
            gs = new_gs()
            lrd = tf.train.inverse_time_decay(multiplier * self.oo_lr[0], gs, self.oo_lr[1],
                                              self.oo_lr[2], name='o_lrd')
            return opt_f(lrd)
        else:
            raise AttributeError('not understood')


class KNNLDSConfig(LDSConfig):
    def __init__(self, method_name='lds', **kwargs):
        """Configuration instance for the method kNN-LDS.
        The arguments are the same as LDSConfig plus

        `k` (10): number of neighbours

        `metric` (cosine): metric function to use"""
        self.k = 10
        self.metric = 'cosine'
        super().__init__(method_name, **kwargs)


def lds(data_conf: ConfigData, config: LDSConfig, return_out=False):
    """
    Runs the LDS algorithm on data specified by `data_conf` with parameters
    specified in `config`.

    :param data_conf: Configuration for the data. Please see `ConfigData` for documentation
    :param config: Configuration for the method's parameters. Please see `LDSConfig` for documentation
    :return: a triplet: - the local variable dictionary (as returned by vars()),
                        - the best `early stopping` accuracy
                        - the test accuracy on the iteration that achieved the best `early stopping` accuracy
    """
    ss = setup_tf(config.seed)

    adj, adj_mods, features, ys, train_mask, val_mask, es_mask, test_mask, class_labels = data_conf.load()
    plc = Placeholders(features, ys)

    if isinstance(config, KNNLDSConfig):
        g = kneighbors_graph(features, config.k, metric=config.metric)
        g = np.array(g.todense(), dtype=np.float32)
        adj_mods = g

    constraint = upper_tri_const(adj.shape)
    adj_hyp = get_stc_hyperparameter('adj_hyp',
                                    constraint(adj_mods + config.est_p_edge * tf.ones(adj.shape)),
                                    constraints=constraint, sample_func=symm_adj_sample)

    out, ws, rep = dense_gcn_2l_tensor(plc.X, adj_hyp, plc.Y, num_layer=config.num_layer, dropout=plc.keep_prob)

    #error = tf.identity(gcn.metrics.masked_softmax_cross_entropy(out, plc.Y, plc.label_mask), 'error')
    error = tf.identity(gcn.metrics.masked_mse(out, plc.Y, plc.label_mask),'error')
    tr_error = error + config.l2_reg * tf.nn.l2_loss(ws[0])

    acc = tf.identity(gcn.metrics.masked_accuracy(out, plc.Y, plc.label_mask), 'accuracy')

    tr_fd, val_fd, es_fd, test_fd = plc.fds(train_mask, val_mask, es_mask, test_mask)
    tr_fd = {**tr_fd, **{plc.keep_prob: config.keep_prob}}

    svd = init_svd(data_conf, config)  # initialize data structure for saving statistics and perform early stopping

    def _test_on_accept(_t):  # execute this when a better parameter configuration is found
        accft = empirical_mean_model(config.n_sample, adj_hyp, out, error, fd=test_fd) #change acc with error
        update_append(svd, oa_t=_t, oa_act=accft)
        print('iteration', _t, ' - mean test accuracy: ', accft)
        global test_acc
        test_acc = accft

    # initialize hyperparamter optimization method
    ho_mod = far.HyperOptimizer(StcReverseHG())
    io_opt, oo_opt = config.io_optimizer(), config.oo_optimizer()
    ho_step = ho_mod.minimize(error, oo_opt, tr_error, io_opt, global_step=get_gs())

    # run the method
    tf.global_variables_initializer().run()
    es_gen = early_stopping_with_save(config.pat, ss, svd, on_accept=_test_on_accept)

    if config.train:
        try:
            for _ in es_gen:  # outer optimization loop
                e_es, a_es = empirical_mean_model(config.n_sample, adj_hyp, out, error, acc, fd=es_fd) #remove acc after error

                es_gen.send(-e_es)  # new early stopping accuracy

                # records some statistics -------
                etr, atr = ss.run([error, acc], tr_fd)
                print('Training Error:', etr)
                eva, ava = ss.run([error, acc], val_fd)
                print('Val Error:', eva)
                ete, ate = ss.run([error, acc], test_fd)
                print('Test Error:', ete)
                iolr = ss.run(io_opt.optimizer_params_tensor)
                n_edgs = np.sum(adj_hyp.eval())

                update_append(svd, etr=etr, atr=atr, eva=eva, ava=ava, ete=ete, ate=ate, iolr=iolr,
                              e_es=e_es, e_ac=a_es, n_edgs=n_edgs, olr=ss.run('o_lrd:0'))
                # end record ----------------------

                steps, rs = ho_mod.hypergradient.min_decrease_condition(
                    config.io_params[0], config.io_params[1], config.io_steps,
                    feed_dicts=tr_fd, verbose=False, obj=None)

                for j in range(config.io_params[2] // config.io_steps):  # inner optimization loop
                    if rs['pat'] == 0: break  # inner objective is no longer decreasing
                    ho_step(steps(), tr_fd, val_fd, online=j)  # do one hypergradient optimization step

        except KeyboardInterrupt:
            print('Interrupted.', file=sys.stderr)
            return vars()
    
    if return_out:
        test_preds = ss.run(fetches=out, feed_dict=test_fd)[test_mask]
        print('test_preds shape:', test_preds.shape)
        return vars(), svd['es final value'], test_acc, test_preds

    return vars(), svd['es final value'], test_acc


def KNN(X, y, n):
    l = len(y)
    y_hat = []
    for i in range(l):
        X_train = np.delete(X, i, axis = 0)
        y_train = np.delete(y, i, axis = 0)
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
    acc = sum(np.array(y_hat) == y) / l
    return acc


def test_KNN(pred, labels, y, idx_train, idx_test, idx_val, test_on_val=False, N=[10]):
    '''to test for train+valid set, provide idx_train and set idx_test=idx_valid''' 
    if not test_on_val: #evaluating on train+val+test set, pred=test_preds
        X = np.r_[labels[idx_train], labels[idx_val], pred]
        y = np.r_[y[idx_train], y[idx_val], y[idx_test]]
    else: #evaluating on train+val set
        X = np.r_[labels[idx_train], pred] #pred=val_preds
        y = np.r_[y[idx_train], y[idx_val]]
    
    print(X.shape)
    print(y.shape)

    result = []
    for n in N: # classification n_neighbors = 5, n_components = 30 up
        result.append(KNN(X.copy(), y.copy(), n))
    
    return result


def main(data, method, seed, missing_percentage):

    if data == 'breast_cancer':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=True)

    if data in imputation_datasets:
        data_config = ImputationDataset(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=True)

    if method == 'knnlds':
        configs = KNNLDSConfig.grid(pat=20,
                                    seed=seed, io_steps=[5, 16],
                                    keep_prob=[0.5, 0.75 , 1.0],
                                    io_lr=[1e-3, 1e-4],
                                    oo_lr=[1e-3, 1e-4],
                                    metric=['cosine', 'minkowski'], k=[10, 20])
    best_valid_acc = -math.inf
    best_test_acc = -math.inf
    best_preds = None

    adj, adj_mods, features, ys, train_mask, val_mask, es_mask, test_mask, class_labels = data_config.load()

    print(data_config)
    for cnf in configs:
        print(cnf)
        vrs, valid_acc, test_acc, test_preds = lds(data_config, cnf, return_out=True)
        if best_valid_acc <= valid_acc:
            print('Found a better configuration:', valid_acc)
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            best_preds = test_preds
        print('Test accuracy of the best found model:', best_test_acc)
        
        print('INTERMEDIATE KNN RESULT: ')
        knn_result = test_KNN(test_preds, ys, class_labels, train_mask, test_mask, val_mask, test_on_val=False, N=[10])
        print(knn_result)
    
    print('BEST KNN RESULT: ')
    knn_result = test_KNN(best_preds, ys, class_labels, train_mask, test_mask, val_mask, test_on_val=False, N=[10])
    print(knn_result)
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='method')
    parser.add_argument('-d', default='breast_cancer', type=str,
                        help='The evaluation dataset: iris, wine, breast_cancer, digits, 20newstrain, 20news10, ' +
                        'cora, citeseer, fma. Default: breast_cancer')
    parser.add_argument('-m', default='knnlds', type=str,
                        help='The method: lds or knnlds. Default: knnlds')
    parser.add_argument('-s', default=1, type=int,
                        help='The random seed. Default: 1')
    parser.add_argument('-e', default=50, type=int,
                        help='The percentage of missing edges (valid only for cora and citeseer dataset): Default 50. - ' +  
                        'PLEASE NOTE THAT the x-axes of Fig. 2  in the paper reports the percentage of retained edges rather ' +
                        'than that of missing edges.')
    args = parser.parse_args()

    _data, _method, _seed, _missing_percentage = args.d, args.m, args.s, args.e/100

    main(_data, _method, _seed, _missing_percentage)
