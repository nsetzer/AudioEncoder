
import json
import glob
import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array_feature(array):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))

def reshape(x, shape):
    """shape can be a mixed set of Tensors or Integers"""
    return tf.reshape(x, tf.stack(shape))

def unison_shuffled_copies(a, b):
    """
    creates a data copy of both input arrays
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

class DatasetWriter(object):
    """docstring for DatasetWriter"""
    def __init__(self, path):
        super(DatasetWriter, self).__init__()

        self.path = path

        self.metadata = {
            "feature": {
                "shape": None,
                "type": "float",
            },
            "label": {
                "shape": None,
                "type": "int8",
            }
        }

        self.count = 0

    def open(self):
        self.writer = tf.python_io.TFRecordWriter(self.path)

    def close(self):

        self.writer.close()

    def addGrayscaleImage(self, x, y, uid=None):
        """
        x: 2d array of floats
        y: onehot encoding of class
        """

        x = x.ravel()
        y = y.astype(np.int8)

        feature_map = {
            "label": _bytes_feature(y.tostring()),
            "feat": _float_array_feature(x),
        }

        if uid is not None:
            feature_map['uid'] = _bytes_feature(uid.encode("utf-8"))

        feature = tf.train.Features(feature=feature_map)

        example = tf.train.Example(features=feature)

        self.writer.write(example.SerializeToString())

        self.count += 1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

class DatasetReader(object):
    """docstring for DatasetReader"""
    def __init__(self):
        super(DatasetReader, self).__init__()

class Dataset(object):

    def __init__(self):
        super(Dataset, self).__init__()

        self.train_path = None
        self.dev_path = None
        self.test_path = None

        self.iterTrain = None
        self.iterDev = None
        self.iterTest = None

    def shape(self, batch_size, flat=False):
        if flat:
            return [batch_size, self.feat_width * self.feat_height]
        return [batch_size, self.feat_width, self.feat_height]

    def getTrain(self, batch_size=1, seed=None):

        if self.train_path is None:
            raise Exception(self.train_path)

        if self.iterTrain is None:

            self.dataset_train = newDataset(self.train_path,
                seed=seed, batch_size=batch_size, shuffle_size=1000,
                isTraining=True)

            self.iterTrain = self.dataset_train.make_initializable_iterator()
            self.iterTrainData = self.iterTrain.get_next()

        return self.iterTrainData

    def getDev(self, batch_size=1, seed=None):

        if self.dev_path is None:
            raise Exception(self.dev_path)

        if self.iterDev is None:
            self.dataset_dev = newDataset(self.dev_path,
                seed=seed, batch_size=batch_size, shuffle_size=1000,
                isTraining=True)

            self.iterDev = self.dataset_dev.make_initializable_iterator()
            self.iterDevData = self.iterDev.get_next()

        return self.iterDevData

    def getTest(self, batch_size=1):

        if self.test_path is None:
            raise Exception(self.test_path)

        if self.iterTest is None:
            self.dataset_test = newDataset(self.test_path,
                batch_size=batch_size, shuffle_size=1000,
                isTraining=True)

            self.iterTest = self.dataset_test.make_initializable_iterator()
            self.iterTestData = self.iterTest.get_next()

        return self.iterTestData

    def initializer(self):

        op = []
        if self.iterTrain is not None:
            op.append(self.iterTrain.initializer)
        if self.iterDev is not None:
            op.append(self.iterDev.initializer)
        if self.iterTest is not None:
            op.append(self.iterTest.initializer)
        return op

    def oneHot2Label(self, y):
        raise NotImplementedError()

def _example_parser(example):

    proto = {
        'feat': tf.VarLenFeature(tf.float32),
        'label': tf.FixedLenFeature([], tf.string),
        'uid': tf.VarLenFeature(tf.string),
    }

    sample = tf.parse_single_example(
        example,
        features=proto
    )

    label = sample['label']
    image = sample['feat']
    uid   = sample['uid']

    # image = tf.decode_raw(image, tf.int8)

    label = tf.decode_raw(label, tf.int8)
    label = tf.cast(label, tf.float32)  # TODO: keep this?

    image = tf.sparse_tensor_to_dense(image, default_value=0)

    uid = tf.sparse_tensor_to_dense(uid, default_value=b"")

    return image, label, uid

def _file_parser(filename):
    return tf.data.TFRecordDataset([filename]) \
            .map(_example_parser)

def newDataset(file_pattern, seed=None, isTraining=True,
    cycle_length=None, block_length=10,
    shuffle_size=1000, batch_size=1, prefetch_size=10):
    """
    file_pattern: glob string matching a set of files
    """

    dataset = tf.data.Dataset().list_files(file_pattern)
    if cycle_length is None:
        cycle_length = len(glob.glob(file_pattern))
        print(cycle_length)
    dataset = dataset.interleave(_file_parser,
        cycle_length=cycle_length,block_length=block_length)

    if isTraining:
        dataset = dataset.shuffle(buffer_size=shuffle_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset