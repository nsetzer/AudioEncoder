"""
https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
"""

import tensorflow as tf
import numpy as np

# from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.layers import fully_connected as fc

def vae_fn(x, dimensions, reuse=False):

    # todo: use dimensions
    # todo: n_z is just the last element in dims...
    # todo: pass feature shape to this function

    # for sigmoid activation,
    # assume x is zero mean, between -1 and 1
    # convert to range o..1
    x = tf.add(tf.multiply(x, .5), 0.5)
    input_dim = dimensions[0]
    x = tf.reshape(x, [-1, input_dim])
    print(x.shape)
    n_z = dimensions[-1]
    # Encode
    # x -> z_mean, z_sigma -> z
    with tf.variable_scope('ENCODER', reuse=reuse):

        f1 = fc(x, 256, reuse=reuse, scope='enc_fc1', activation_fn=tf.nn.elu)
        f2 = fc(f1, 128, reuse=reuse, scope='enc_fc2', activation_fn=tf.nn.elu)
        f3 = fc(f2, 64, reuse=reuse, scope='enc_fc3', activation_fn=tf.nn.elu)
        z_mu = fc(f3, n_z, reuse=reuse, scope='enc_fc4_mu', activation_fn=None)
        z_log_sigma_sq = fc(f3, n_z, reuse=reuse, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        z_prime = tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        z = tf.add(z_mu , z_prime, name="OUTPUT_0")

    # Decode
    # z -> y

    with tf.variable_scope('DECODER', reuse=reuse):

        g1 = fc(z, 64, reuse=reuse, scope='dec_fc1', activation_fn=tf.nn.elu)
        g2 = fc(g1, 128, reuse=reuse, scope='dec_fc2', activation_fn=tf.nn.elu)
        g3 = fc(g2, 256, reuse=reuse, scope='dec_fc3', activation_fn=tf.nn.elu)
        y = fc(g3, input_dim, reuse=reuse, scope='OUTPUT_0',
            activation_fn=tf.sigmoid)

    # Loss
    # Reconstruction loss
    # Minimize the cross-entropy loss
    # H(x, y) = -\Sigma x*log(y) + (1-x)*log(1-y)
    epsilon = 1e-10
    recon_loss = -tf.reduce_sum(
        x * tf.log(epsilon+y) + (1-x) * tf.log(epsilon+1-y),
        axis=1
    )
    recon_loss = tf.reduce_mean(recon_loss)

    # Latent loss
    # Kullback Leibler divergence: measure the difference between two distributions
    # Here we measure the divergence between the latent distribution and N(0, 1)
    latent_loss = -0.5 * tf.reduce_sum(
        1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq), axis=1)
    latent_loss = tf.reduce_mean(latent_loss)

    total_loss = tf.reduce_mean(recon_loss + latent_loss)

    return {'x': x, 'z': z, 'y': y, 'cost': total_loss}
