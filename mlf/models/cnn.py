
from .model import Model
import tensorflow as tf
import numpy as np

from functools import reduce
import operator

def prod(seq):
    return reduce(operator.mul, seq, 1)

class cnn(Model):
    # https://www.tensorflow.org/tutorials/layers
    def __init__(self, **kwargs):
        super(cnn, self).__init__(**kwargs)

    def defaultSettings(self):
        settings = {
            "batch_size": None,
            "nFeatures": None,
            "nSlices": None,
            "nClasses": 10
        }
        return settings

    def __call__(self, x, y, reuse=False, isTraining=False):
        # features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        # we don't know the number of selected frames

        cfg = self.settings
        # shape := [batch_size, height, width, 1]
        shape = [-1, cfg['nFeatures'], cfg['nSlices'], 1]
        input_layer = tf.reshape(x, shape)

        print("shape", shape)
        with tf.variable_scope('CONV1', reuse=reuse):
            conv1 = tf.layers.conv2d(
              inputs=input_layer,
              filters=32,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=[2, 2], strides=2)

        with tf.variable_scope('CONV2', reuse=reuse):
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        with tf.variable_scope('DENSE', reuse=reuse):
            # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            # flatten the previous layer for mapping into the dense layer
            shape = [-1, prod(pool2.shape[1:])]
            print(pool2.shape, shape)
            pool2_flat = tf.reshape(pool2, shape)
            dense = tf.layers.dense(
                inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
              inputs=dense, rate=0.4, training=isTraining)

        # Logits Layer
        with tf.variable_scope('LOGITS', reuse=reuse):
            logits = tf.layers.dense(inputs=dropout, units=cfg['nClasses'])

            classes = tf.argmax(input=logits, axis=1)
            probabilities = tf.nn.softmax(logits, name="softmax_tensor")

            # Calculate Loss (for both TRAIN and EVAL modes)
            #loss = tf.losses.sparse_softmax_cross_entropy(
            #    labels=y, logits=logits)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=y))

        # accuracy = tf.metrics.accuracy(
        #  labels=y, predictions=classes)

        # tf.estimator.EstimatorSpec(mode=mode,
        # loss=loss, train_op=train_op)

        # tf.estimator.EstimatorSpec(
        # mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        ops = {
            "x": x,
            "y": y,
            "logits": logits,
            "prediction": probabilities,
            "classes": classes,
            "cost": loss,
        }
        return ops
