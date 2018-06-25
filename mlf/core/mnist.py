
import os
import sys

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import DatasetWriter, DatasetReader, Dataset, \
    reshape, unison_shuffled_copies


class MnistDataset(Dataset):
    """

        assume rectangular features with known dimensions
        assume labels are one-hot encoding
    """
    def __init__(self, data_dir, classes=None):
        super(MnistDataset, self).__init__()
        self.data_dir = data_dir

        self.classes = list(range(10)) if classes is None else classes

        self.train_path = os.path.join(self.data_dir,
            "train-%d.record" % len(self.classes))

        self.dev_path = os.path.join(self.data_dir,
            "dev-%d.record" % len(self.classes))

        self.test_path = os.path.join(self.data_dir,
            "test-%d.record" % len(self.classes))

        if not os.path.exists(self.train_path):
            self._create()

        self.feat_width = 28
        self.feat_height = 28

    def _create(self):

        nClasses = len(self.classes)

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

        print(len(self.classes), train_data.shape, train_labels.shape)
        # select the classes to train on (default all)
        if nClasses != 10:
            indices = np.hstack([np.where(train_labels==c) for c in self.classes])
            train_labels = train_labels[indices][0]
            train_data = train_data[indices][0]
            print(train_data.shape, train_labels.shape)

        #shuffle training classes
        train_data, train_labels = unison_shuffled_copies(train_data, train_labels)

        print(train_data.shape)
        mean = np.mean(train_data)
        train_data = (train_data - mean)
        _min = np.min(train_data)
        _max = np.max(train_data)
        train_data = train_data / max(abs(_min), abs(_max))
        _var = np.var(train_data)
        _std = np.std(train_data)
        print("data mean", _min, mean, _max, _var, _std)

        # create a 1-hot encoding mapping
        eye = []
        for i in range(10):
            row = [0] * nClasses
            if nClasses == 10:
                row[i] = 1
            elif i in self.classes:
                row[self.classes.index(i)] = 1
            eye.append(row)
        eye = np.asarray(eye)

        # convert class to a one-hot encoding
        train_labels = eye[train_labels]

        test_data = mnist.test.images  # Returns np.array

        mean = np.mean(test_data)
        test_data = (test_data - mean)
        _min = np.min(test_data)
        _max = np.max(test_data)
        test_data = test_data / max(abs(_min), abs(_max))
        _var = np.var(test_data)
        _std = np.std(test_data)
        print("data mean", _min, mean, _max, _var, _std)

        test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        if nClasses != 10:
            indices = np.hstack([np.where(test_labels==c) for c in self.classes])
            test_labels = test_labels[indices][0]
            test_data = test_data[indices][0]

        test_labels = eye[test_labels]

        N = len(test_data)
        train_data, dev_data = train_data[N:], train_data[:N]
        train_labels, dev_labels = train_labels[N:], train_labels[:N]

        with DatasetWriter(self.train_path) as w:
            for x,y in zip(train_data, train_labels):
                w.addGrayscaleImage(x, y)

        with DatasetWriter(self.dev_path) as w:
            for x,y in zip(dev_data, dev_labels):
                w.addGrayscaleImage(x, y)

        with DatasetWriter(self.test_path) as w:
            for x,y in zip(test_data, test_labels):
                w.addGrayscaleImage(x, y)

    def oneHot2Label(self, y):
        index = np.where(np.asarray(y) == 1)[0][0]
        return self.classes[index]