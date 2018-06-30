"""
https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
https://github.com/shaohua0116/VAE-Tensorflow
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
    #x = tf.add(tf.multiply(x, .5), 0.5)

    batch_size=-1
    x = tf.reshape(x, [batch_size, dimensions[0]])

    # Encode
    # x -> z_mean, z_sigma -> z
    with tf.variable_scope('ENCODER', reuse=reuse):
        # input, 512, 256, 128
        # x, 512 -> x, 256 -> x, 128
        c = x
        for i, dim in enumerate(dimensions[1:-1]):
            c = fc(c, dim, reuse=reuse, scope='enc_fc_%d' % i, activation_fn=tf.nn.elu)
        z_mu = fc(c, dimensions[-1], reuse=reuse, scope='enc_fc_mu', activation_fn=None)
        z_log_sigma_sq = fc(c, dimensions[-1], reuse=reuse, scope='enc_fc_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        z_prime = tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        z = tf.add(z_mu , z_prime, name="OUTPUT_0")

    # Decode
    # z -> y

    with tf.variable_scope('DECODER', reuse=reuse):

        c = z
        for i, dim in enumerate(reversed(dimensions[:-1])):
            c = fc(c, dim, reuse=reuse, scope='dec_fc1', activation_fn=tf.nn.elu)
        y = fc(c, dimensions[0], reuse=reuse, scope='OUTPUT_0',
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
