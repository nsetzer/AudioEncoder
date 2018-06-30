

import tensorflow as tf
import numpy as np

from .model import Model

class autoencoder(Model):
    """docstring for autoencoder"""
    def __init__(self, **kwargs):
        super(autoencoder, self).__init__(**kwargs)

    def defaultSettings(self):
        settings = {
            # a list of dimensions for the autoencoder
            # first dimension is the input shape of the data
            # last dimensions is the shape of the latent layer
            "dimensions": None
        }
        return settings;

    def __call__(self, x, y, reuse=False, isTraining=False):
        """Build a deep autoencoder w/ tied weights.
        Parameters
        ----------
        dimensions : list, optional
            The number of neurons for each layer of the autoencoder.
        Returns
        -------
        x : Tensor
            Input placeholder to the network
        z : Tensor
            Inner-most latent representation
        y : Tensor
            Output reconstruction of the input
        cost : Tensor
            Overall cost to use for training
        """
        # %% input to the network

        current_input = x

        encoder_dims = dimensions = self.settings['dimensions']
        decoder_dims = encoder_dims[::-1]

        # %% Build the encoder
        encoder = []
        with tf.variable_scope('ENCODER', reuse=reuse):

            for layer_i in range(1, len(encoder_dims)):
                layer_prev = layer_i - 1
                n_input = encoder_dims[layer_prev]
                n_output = encoder_dims[layer_i]
                W = tf.get_variable("weight_%d" % layer_i,
                    initializer=tf.random_uniform([n_input, n_output],
                        -1.0 / np.sqrt(n_input),
                         1.0 / np.sqrt(n_input)))
                b = tf.get_variable("encoder_bias_%d" % layer_i,
                    initializer=tf.zeros([n_output]))
                encoder.insert(0, W)
                # name is selected so that output 0 is always
                # the latent representation
                name = "OUTPUT_%d" % (len(encoder_dims) - layer_i - 1)
                output = tf.nn.tanh(tf.matmul(current_input, W) + b,
                    name=name)
                print("%s: %dx%d" % (output.name, n_input, n_output))
                current_input = output

        # Latent representation (embedding, neural coding)
        z = current_input

        with tf.variable_scope('DECODER', reuse=reuse):

            # Build the decoder using the same weights
            for layer_i in range(1, len(decoder_dims)):
                layer_prev = layer_i - 1
                n_input = decoder_dims[layer_prev]
                n_output = decoder_dims[layer_i]
                W = tf.transpose(encoder[layer_i-1])
                b = tf.get_variable("decoder_bias_%d" % layer_i,
                    initializer=tf.zeros([n_output]))
                # name is selected so that output 0 is always
                # the reconstructed signal
                name = "OUTPUT_%d" % (len(decoder_dims) - layer_i - 1)
                output = tf.nn.tanh(tf.matmul(current_input, W) + b,
                    name=name)
                print("%s: %dx%d" % (output.name, n_input, n_output))

                current_input = output

        # Now have the reconstruction through the network
        y = current_input

        # Cost function measures pixel-wise difference
        cost = tf.reduce_sum(tf.square(y - x), name="COST")

        return {'x': x, 'z': z, 'y': y, 'cost': cost}