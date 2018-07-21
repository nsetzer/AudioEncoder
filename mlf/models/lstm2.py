
from .model import Model
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

class lstm2(Model):
    def __init__(self, **kwargs):
        super(lstm2, self).__init__(**kwargs)

    def defaultSettings(self):
        settings = {
            "nClasses": None,
            "batch_size": None,
            "nFeatures": None,
            "nSlices": None,
            "num_hidden": 512,
        }
        return settings

    def __call__(self, x, y, reuse=False, isTraining=False):
        # https://colab.research.google.com/drive/18FqI18psdH30WUJ1uPd6zVgK2AwxO_Bj#scrollTo=GyGSqEoqa7r2

        cfg = self.settings
        # shape := [batch_size, height, width, 1]
        shape = [-1, cfg['nFeatures'], cfg['nSlices']]
        input_layer = tf.reshape(x, shape)

        with tf.variable_scope('RNN', reuse=reuse):
            # Define a lstm cell with tensorflow
            lstm_cell = rnn.LSTMBlockCell(
                self.settings['num_hidden'], forget_bias=1.0)

            # Get lstm cell output
            # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(
                cell=lstm_cell, inputs=input_layer, time_major=False, dtype=tf.float32)

        with tf.variable_scope('LOGITS', reuse=reuse):

            output_layer = tf.layers.Dense(
                self.settings['nClasses'], activation=None,
                kernel_initializer=tf.orthogonal_initializer()
            )

            logits = output_layer(tf.layers.batch_normalization(outputs[:, -1, :]))

            classes = tf.argmax(input=logits, axis=1)
            probabilities = tf.nn.softmax(logits, name="softmax_tensor")

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits, labels=y))

        ops = {
            "x": x,
            "y": y,
            "logits": logits,
            "prediction": probabilities,
            "classes": classes,
            "cost": loss,
        }

        return ops

    def __x__call__(self, x, y, reuse=False, isTraining=False):

        n_classes = self.settings['nClasses']
        n_hidden = self.settings['nRecurrentUnits']

        # TODO: pass in width and height
        x = tf.reshape(x, [self.settings['batch_size'],
            self.settings['height'],
            self.settings['width']])

        wshape = [n_hidden, n_classes]
        weight = tf.Variable(tf.truncated_normal(wshape, stddev=0.1))

        bshape = [n_classes]
        bias = tf.Variable(tf.constant(0.0, shape=bshape))

        cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        output, state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype=tf.float32)
        output_flattened = tf.reshape(output, [-1, n_hidden])
        output_logits = tf.add(tf.matmul(output_flattened, weight), bias)
        output_all = tf.nn.sigmoid(output_logits)
        output_reshaped = tf.reshape(output_all, [-1, n_steps, n_classes])
        output_last = tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), n_steps - 1)
        #output = tf.transpose(output, [1, 0, 2])
        #last = tf.gather(output, int(output.get_shape()[0]) - 1)
        #output_last = tf.nn.sigmoid(tf.matmul(last, weight) + bias)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=output_logits, labels=y))

        ops = {
            "x": x,
            "y": y,
            "logits": output_logits,
            "prediction": output_last,
            "classes": output_all,
            "cost": cost,
        }
        return ops

