
from .model import Model
import tensorflow as tf
import numpy as np

class lstm2(Model):
    def __init__(self, **kwargs):
        super(lstm2, self).__init__(**kwargs)

    def defaultSettings(self):
        settings = {
            "nClasses": None,
            "nRecurrentUnits": 100,
            "batch_size": None,
            "width": 28,  # should be none
            "height": -1,
        }
        return settings

    def __call__(self, x, y, reuse=False, isTraining=False):

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

