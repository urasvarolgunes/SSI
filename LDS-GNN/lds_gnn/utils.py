"""Utility methods for configurations, logging, early stopping, etc."""

import glob
import gzip
import os
import pickle
import sys
import time
from collections import defaultdict, OrderedDict

import far_ho as far
import numpy as np
import tensorflow as tf


VERSION = 1
SAVE_DIR = 'results'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print('FOLDER', SAVE_DIR, 'CREATED')
SVD_MAP_FILE = '.svd_map'


class Config:
    """ Base class of a configuration instance; offers keyword initialization with easy defaults,
    pretty printing and grid search!
    """
    def __init__(self, **kwargs):
        self._version = 1
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))

    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items())) + ']\n'

    @classmethod
    def default_instance(cls):
        return cls()

    @classmethod
    def grid(cls, **kwargs):
        """Builds a mesh grid with given keyword arguments for this Config class.
        If the value is not a list, then it is considered fixed"""

        class MncDc:
            """This is because np.meshgrid does not always work properly..."""

            def __init__(self, a):
                self.a = a  # tuple!

            def __call__(self):
                return self.a

        sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
        for k, v in sin.items():
            copy_v = []
            for e in v:
                copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
            sin[k] = copy_v

        grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
        return [cls(**far.utils.merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
        )) for vv in grd]


class Placeholders:
    def __init__(self, X, Y):
        self.label_mask = tf.placeholder(tf.int32)
        self.X = tf.constant(X, tf.float32)
        self.Y = tf.constant(Y, tf.float32)
        self.keep_prob = tf.placeholder_with_default(1., shape=())
        self.n = X.shape[0]

    def fd(self, mask, *other_fds):
        return far.utils.merge_dicts({self.label_mask: mask}, *other_fds)

    def fds(self, *masks):
        return [self.fd(m) for m in masks]


class GraphKeys(far.GraphKeys):
    STOCHASTIC_HYPER = 'stochastic_hyper'


def setup_tf(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    if tf.get_default_session(): tf.get_default_session().close()
    return tf.InteractiveSession()


def new_gs():  # creator for the global step
    return tf.Variable(0, trainable=False, name='step',
                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])


def get_gs():
    try:
        return tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
    except IndexError:
        return None


def upper_triangular_initializer(const):
    def _init(shape, dtype, _=None):
        a = np.zeros(shape)
        for i in range(0, shape[0]):
            for j in range(i + 1, shape[1]):
                a[i, j] = const
        return tf.constant(a, dtype=dtype)

    return _init


def upper_tri_const(shape, minval=0., maxval=1.):
    return lambda v: tf.maximum(tf.minimum(v * upper_triangular_mask(shape), maxval), minval)


def box_const(minval=0., maxval=1.):
    return lambda v: tf.maximum(tf.minimum(v, maxval), minval)


def upper_triangular_mask(shape, as_array=False):
    a = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(i + 1, shape[1]):
            a[i, j] = 1.
    return tf.constant(a, dtype=tf.float32) if not as_array else a


def init_svd(data_config=None, config=None):
    fn = ''
    if data_config: fn += str(data_config)
    if config: fn += str(config)
    return defaultdict(list, (('config', config), ('data_config', data_config),
                              ('name', fn)))


def restore_from_svd(svd, session=None, verbose=False):
    if session is None: session = tf.get_default_session()
    for v in tf.global_variables():
        try:
            session.run(v.assign(svd[v.name]))
            if verbose: print(v.name, 'restored')
        except KeyError:
            print('WARNING: variable', v, 'not in SVD', file=sys.stderr)
        except ValueError as e:
            print(e, file=sys.stderr)


def update_append(dct, **updates):
    for k, e in updates.items():
        dct[k].append(e)


def update_append_v2(dct: dict, upd_dct: dict):
    for k, e in upd_dct.items():
        dct[k].append(e)


def gz_read(name, results=True):
    name = '{}/{}.gz'.format(SAVE_DIR, name) if results else '{}.gz'.format(name)
    with gzip.open(name, 'rb') as f:
        return pickle.load(f)


def gz_write(content, name, results=True):
    name = '{}/{}.gz'.format(SAVE_DIR, name) if results else '{}.gz'.format(name)
    with gzip.open(name, 'wb') as f:
        pickle.dump(content, f)


def list_results(*keywords, verbose=True):
    _strip_fn = lambda nn: nn.split(os.sep)[1][:-3]

    svd_map = gz_read(SVD_MAP_FILE, results=False)
    result_list = sorted(glob.glob('{}/*.gz'.format(SAVE_DIR)),
                         key=lambda x: os.path.getmtime(x))
    result_list = map(_strip_fn, result_list)
    try:
        result_list = list(filter(lambda nn: all([kw in svd_map[nn] for kw in keywords]), result_list))
    except KeyError:
        print('Misc.list_results: something wrong happened: returning None', file=sys.stderr)
        return []
    if verbose:
        for k, v in enumerate(result_list):
            print(k, '->', svd_map[v])
    return result_list


# noinspection PyUnboundLocalVariable
def load_results(*keywords, exp_id=None, verbose=True):
    if verbose: print('loading results:')
    rs = list_results(*keywords, verbose=verbose)
    if exp_id is not None:
        rs = [rs[exp_id]]
    ldd = list(map(gz_read, rs))
    loaded = ldd if len(ldd) > 1 else ldd[0]
    return loaded


def early_stopping(patience, maxiters=1e10, on_accept=None, on_refuse=None, on_close=None, verbose=True):
    """
    Generator that implements early stopping. Use `send` method to give to update the state of the generator
    (e.g. with last validation accuracy)

    :param patience:
    :param maxiters:
    :param on_accept: function to be executed upon acceptance of the iteration
    :param on_refuse: function to be executed when the iteration is rejected (i.e. the value is lower then best)
    :param on_close: function to be exectued when early stopping activates
    :param verbose:
    :return: a step generator
    """
    val = None
    pat = patience
    t = 0
    while pat and t < maxiters:
        new_val = yield t
        if new_val is not None:
            if val is None or new_val > val:
                val = new_val
                pat = patience
                if on_accept:
                    try:
                        on_accept(t, val)
                    except TypeError:
                        try:
                            on_accept(t)
                        except TypeError:
                            on_accept()
                if verbose: print('ES t={}: Increased val accuracy: {}'.format(t, val))
            else:
                pat -= 1
                if on_refuse: on_refuse(t)
        else:
            t += 1
    yield
    if on_close: on_close(val)
    if verbose: print('ES: ending after', t, 'iterations')


def early_stopping_with_save(patience, ss, svd, maxiters=1e10, var_list=None,
                             on_accept=None, on_refuse=None, on_close=None, verbose=True):
    starting_time = -1
    gz_name = str(time.time())
    svd['file name'] = gz_name

    def _on_accept(t, val):
        nonlocal starting_time
        if starting_time == -1: starting_time = time.time()
        _var_list = tf.global_variables() if var_list is None else var_list

        svd.update((v.name, ss.run(v)) for v in _var_list)
        svd['on accept t'].append(t)
        if on_accept:
            try:
                on_accept(t, val)
            except TypeError:
                try:
                    on_accept(t)
                except TypeError:
                    on_accept()

    def _on_close(val):
        svd['es final value'] = val
        svd['version'] = VERSION
        svd['running time'] = time.time() - starting_time
        gz_write(svd, gz_name)

        try:
            fl_dict = gz_read(SVD_MAP_FILE, results=False)
        except FileNotFoundError:
            fl_dict = {}
            print('CREATING SVD MAP FILE WITH NAME:', SVD_MAP_FILE)

        fl_dict[gz_name] = svd['name']
        gz_write(fl_dict, SVD_MAP_FILE, results=False)

        if on_close:
            try:
                on_close(val)
            except TypeError:
                on_close()

    return early_stopping(patience, maxiters,
                          on_accept=_on_accept,
                          on_refuse=on_refuse,
                          on_close=_on_close,
                          verbose=verbose)
