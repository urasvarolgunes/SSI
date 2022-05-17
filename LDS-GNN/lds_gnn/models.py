"""Implements a dense version of a standard GCN model"""

import tensorflow as tf

ACTIVATION = tf.nn.relu


def dense_gcn_2l_tensor(x, adj, y, hidden_dim=600, num_layer=2, dropout=None, name='GCN'):
    """ Builds up a simple GCN model with dense adjacency matrix and features."""
    f_dim = int(x.shape[1])
    out_dim = int(y.shape[1])

    tilde_A = adj + tf.eye(int(adj.shape[0]))
    tilde_D = tf.reduce_sum(tilde_A, axis=1)
    sqrt_tildD = 1. / tf.sqrt(tilde_D)
    daBda = lambda _b, _a: tf.transpose(_b * tf.transpose(_a)) * _b
    hatA = daBda(sqrt_tildD, tilde_A)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = []
        for i in range(num_layer):
            if i == 0:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(f_dim, hidden_dim)))
            elif i == num_layer - 1:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, out_dim)))
            else:
                W.append(tf.get_variable('W%s' % i, initializer=tf.glorot_uniform_initializer(),
                                         shape=(hidden_dim, hidden_dim)))

    representation = ACTIVATION(hatA @ x @ W[0])
    if dropout is not None:
        representation = tf.nn.dropout(representation, dropout)
    for i in range(1, num_layer - 1):
        representation = ACTIVATION(hatA @ representation @ W[i])
        if dropout is not None:
            representation = tf.nn.dropout(representation, dropout)
    out = tf.identity(hatA @ representation @ W[-1], 'out')

    return out, W, representation
