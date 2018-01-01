import numpy as np
import tensorflow as tf

#### This module define network architectures using algorithmic/data params

def autoencoder_ops(weight_params_inits, data_params, bias_trainable=True,
         init_weights=None, init_bias=None):
    width, activation_fn, norm = weight_params_inits
    data_dim = data_params['data_dim']
    train_batch_size = data_params['train_batch_size']
    ## parameter initialization
    if init_weights is None:
        init_weights = tf.random_normal([width, data_dim], dtype=tf.float64) # init random weights
        with tf.Session() as sess:
            init_weights_ = sess.run(init_weights)
    else:
        init_weights_ = init_weights

    if init_bias is None:
        init_bias = np.zeros(width)
        with tf.Session() as sess:
            init_bias_ = sess.run(init_bias)
    else:
        init_bias_ = init_bias

    ## parameter defintiion
    weights = tf.Variable(init_weights_, dtype=tf.float64)
    bias = tf.Variable(init_bias_, trainable=bias_trainable, dtype=tf.float64)

    ## data format
    x = tf.placeholder(tf.float64, [train_batch_size, data_dim])
    ## encoder
    hidden = tf.add(tf.matmul(x, tf.transpose(weights)), bias) ## train_batch_size by width
    ## decoder
    if activation_fn == 'relu':
        # train_batch_size by data_dim
        xhat = tf.squeeze(tf.matmul(tf.nn.relu(hidden), weights))
    return weights, init_weights_, bias, init_bias_, xhat, x
