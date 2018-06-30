#! cd .. && python -m mlf.experiment

import os
import sys
import shutil
import logging

from .core.mnist import MnistDataset
from .core.dataset import Dataset
from .core.config import AutoEncoderConfig
from .models.autoencoder import autoencoder
from .models.vae import vae_fn
from .tools.freeze import freeze
# from .models.rnn import lstm

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from SigProc.histogrammer import AutomaticScale
except ImportError as e:
    print(e)

class AudioDataset(Dataset):
    def __init__(self, cfg):
        super(AudioDataset, self).__init__()
        self.cfg = cfg

        # the image was transposed (from the sigproc output)
        # first dimension (column) is a feature
        # second dimension (row) is time
        self.feat_width = self.cfg.featureHeight()
        self.feat_height = self.cfg.sliceSize

        self.train_path = self.cfg.getDatasetGlobPattern("train")
        self.dev_path = self.cfg.getDatasetGlobPattern("dev")
        self.test_path = self.cfg.getDatasetGlobPattern("test")

    def oneHot2Label(self, y):
        index = np.where(np.asarray(y) == 1)[0][0]
        return self.cfg.getGenres()[index]

class TrainerBase(object):

    def __init__(self):
        super(TrainerBase, self).__init__()

    def checkpointExists(self):

        return os.path.exists(os.path.join(
            self.settings['outputDir'], "checkpoint"))

    def restore(self, sess):
        self.saver.restore(sess,
                tf.train.latest_checkpoint(self.settings['outputDir']))

    def beforeSession(self):
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, sess):

        logging.info("training model")

        sess.run(self.init_op)

        tf.train.write_graph(
            sess.graph.as_graph_def(),
            self.settings['outputDir'],
            self.settings['modelFile'],
            as_text=True
        )

        for epoch_i in range(self.settings['nEpochs']):

            sess.run(self.dataset.initializer(),
                   feed_dict={self.seed: epoch_i})

            self.onEpochBegin(sess, epoch_i)

            step = 0
            total_cost = 0
            while True:
                try:
                    _, cost = sess.run([self.train_op,
                        self.train_ops['cost']])
                    total_cost += cost
                    step += 1

                    _step = int(500 / self.settings['batch_size'])
                    _scale = self.dataset.shape(None, flat=True)[-1]

                    _scale = _scale * step * self.settings['batch_size']
                    if step % _step == 0:
                        print( epoch_i, step, total_cost/_scale)

                    if self.settings['max_steps'] > 0 and \
                      step > self.settings['max_steps']:
                        break
                except tf.errors.OutOfRangeError:
                    break

            costs = []
            while True:
                try:
                    costs.append(sess.run(self.dev_ops['cost']))
                except tf.errors.OutOfRangeError:
                    break

            print(epoch_i, np.sum(costs), np.mean(costs),
                np.max(costs), np.min(costs))

            self.saver.save(sess, self.settings['checkpointFile'],
                global_step=epoch_i)

            self.onEpochEnd(sess, epoch_i)

    def export(self, ckpt=None):
        """

        """

        ckpt = ckpt or tf.train.latest_checkpoint(self.settings['outputDir'])

        logging.info("Exporting model using %s" % ckpt)

        tf.reset_default_graph()

        evalOutputDir = os.path.join(self.settings['outputDir'], 'eval')

        if not os.path.exists(evalOutputDir):
            os.makedirs(evalOutputDir)

        init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer(), name="INIT")

        batch_size = 1  # TODO, can this be None or -1 for arbitrary?
        shape = self.dataset.shape(batch_size, flat=True)
        featEval = tf.placeholder(tf.float32, shape, name='INPUT')
        eval_ops = self.model_fn(featEval,
            self.settings['dimensions'], reuse=False)

        # -------------------------------------------------------------------------
        # Export model
        # -------------------------------------------------------------------------
        saver = tf.train.Saver()
        modelFile = 'model.pbtxt'
        modelPath = os.path.join(evalOutputDir, modelFile)
        frozenModelPath = os.path.join(evalOutputDir, "frozen_model.pb")

        model_ckpt = os.path.join(evalOutputDir, 'model.ckpt')
        with tf.Session() as sess:
            sess.run(init_op)

            saver.restore(sess, ckpt)

            saver.save(sess, model_ckpt,
                       write_meta_graph=False)

            tf.train.write_graph(
                sess.graph.as_graph_def(),
                evalOutputDir,
                modelFile,
                as_text=True
            )

        export_ops = list(eval_ops.values())
        export_ops.insert(0,init_op)
        export_names = ','.join([op.name.split(":")[0] for op in export_ops])
        ckpt = tf.train.latest_checkpoint(evalOutputDir)
        print("-"*60)
        print("path: %s" % modelPath)
        print("path: %s" % frozenModelPath)
        print("freeze: %s" % export_names)
        print("checkpint: %s" % ckpt)
        print("-"*60)

        # freeze weights along with the graph, so that it can be used
        # with the C API
        freeze(modelPath, frozenModelPath, ckpt, export_names)

    def onEpochBegin(self, sess, index):
        pass

    def onEpochEnd(self, sess, index):
        pass

class EncoderTrainer(TrainerBase):

    def __init__(self, model_fn):
        super(EncoderTrainer, self).__init__()

        self.model_fn = model_fn

        self.actions = {
            "project": self.project,
            "sample": self.sample,
        }

    def makeGraph(self, settings, dataset):

        self.settings = settings
        self.dataset = dataset

        self.seed = tf.placeholder(tf.int64, shape=tuple(), name="seed")

        logging.info("create training graph")
        featTrain, labelTrain, uidTrain = dataset.getTrain(settings['batch_size'], self.seed)
        self.train_ops = self.model_fn(featTrain, labelTrain,
            reuse=False, isTraining=True)

        logging.info("create dev graph")
        featDev, labelDev, uidDev = dataset.getDev(settings['batch_size'], self.seed)
        self.dev_ops = self.model_fn(featDev, labelDev,
            reuse=True, isTraining=False)

        logging.info("create test graph")
        featTest, labelTest, uidTest = dataset.getTest()
        self.test_ops = self.model_fn(featTest, labelTest,
            reuse=True, isTraining=False)

        logging.info("create optimzer: adam")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(settings['learning_rate'])
        self.train_op = self.optimizer.minimize(self.train_ops['cost'],
            global_step=self.global_step)

    def project(self, sess):

        logging.info("Creating projection")

        self.settings['projectDir'] = os.path.join(self.settings['outputDir'], 'projector')
        self.settings['projectModel'] = os.path.join(self.settings['projectDir'], 'projector.ckpt')

        if not os.path.exists(self.settings['projectDir']):
            os.makedirs(self.settings['projectDir'])

        featTest, labelTest, uidTest = self.dataset.getTest()

        sess.run(self.dataset.initializer(),
                 feed_dict={self.seed: 0})

        embed_op = [self.test_ops['z'], labelTest, uidTest]
        embedded_data = []
        embedded_labels = []
        for i in range(self.settings['n_test_samples']):
            try:
                data, label, uid = sess.run(embed_op)
                # some data sets may not define a uid
                if uid.size > 0:
                    label = uid[0][0].decode("utf-8")
                else:
                    label = self.dataset.oneHot2Label(label[0])
                embedded_data.append(data.reshape([-1, self.settings['dimensions'][-1]]))
                embedded_labels.append(uid)
            except tf.errors.OutOfRangeError:
                    break

        # generate metadata file
        path_metadata = os.path.abspath(
            os.path.join(self.settings['projectDir'], 'metadata.tsv'))

        with open(path_metadata, 'w') as f:
            for label in embedded_labels:
                f.write('%s\n' % label)

        # Input set for Embedded TensorBoard visualization
        # Performed with cpu to conserve memory and processing power
        with tf.device("/cpu:0"):

            # shape must be 2d for tensorboard to parse correctly...
            # TODO: set second dimension correctly
            # TODO: i should not need this reshape if the stack is correct
            stack = tf.stack(np.asarray(embedded_data).reshape([-1, self.settings['dimensions'][-1]]),
                axis=0)

            print("stack shape", stack.shape)

            embedding = tf.Variable(stack, trainable=False,
                name='embedding')

        # saver is required to be created heer...
        saver = tf.train.Saver(max_to_keep=None)
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(self.settings['projectDir'], sess.graph)

        # Add embedding tensorboard visualization. Need tensorflow version
        # >= 0.12.0RC0
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding:0'
        # todo: this should be an absolute path
        embed.metadata_path = path_metadata
        #embed.sprite.image_path = os.path.join(FLAGS.data_dir + '/mnist_10k_sprite.png')

        # Specify the width and height of a single thumbnail.
        #embed.sprite.single_image_dim.extend([28, 28])
        projector.visualize_embeddings(writer, config)

        # We save the embeddings for TensorBoard, setting the global step as
        # The number of data examples
        saver.save(sess, os.path.join(self.settings['projectDir'],
            'a_model.ckpt'),
            global_step=self.settings['n_test_samples'])

    def sample(self, sess, uid=0):

        logging.info("Creating sample image")

        _, width, height = self.dataset.shape(1)

        n = 8
        canvas_orig = np.empty((width * n, height * n))
        canvas_recon = np.empty((width * n, height * n))

        featTest, labelTest, _ = self.dataset.getTest()

        sess.run(self.dataset.initializer(),
                   feed_dict={self.seed: 0})

        for i in range(n):
            for j in range(n):
                # Encode and decode the digit image
                feat, dec = sess.run([featTest, self.test_ops['y']])

                feat_max = feat.max()
                feat_min = feat.min()
                feat = feat.reshape([width, height])
                #feat = AutomaticScale(feat)

                dec_max = dec.max()
                dec_min = dec.min()
                dec = dec.reshape([width, height])
                #dec = AutomaticScale(dec)

                x1 = i * width
                x2 = (i + 1) * width
                y1 = j * height
                y2 = (j + 1) * height
                # Draw the original digits
                canvas_orig[x1:x2, y1:y2] = feat

                # Draw the reconstructed digits
                canvas_recon[x1:x2, y1:y2] = dec

                print(feat_min, feat_max, dec_min, dec_max)

        print("Original Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.savefig(os.path.join(self.settings['outputDir'], "img_%02d_original.png" % uid))

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.savefig(os.path.join(self.settings['outputDir'], "img_%02d_reconstructed.png" % uid))

    def onEpochEnd(self, sess, index):

        self.sample(sess, index)

class ClassifierTrainer(TrainerBase):

    def __init__(self):
        super(ClassifierTrainer, self).__init__()

def train(settings, trainer, dataset):

    trainer.makeGraph(settings, dataset)

    do_export = False

    trainer.beforeSession();
    with tf.Session() as sess:
        if trainer.checkpointExists():
            trainer.restore(sess)
        else:
            trainer.train(sess)
            do_export = True
        trainer.project(sess)

    if do_export:
        trainer.export()

def main():
    logging.basicConfig(level=logging.INFO)

    cfg = AutoEncoderConfig()
    cfg.load("./config/audio_2way.cfg")


    mnist_settings = {
        "dataDir": os.path.abspath("./build/data"),
        "outputDir": os.path.abspath("./build/experiment"),
        "modelFile": "model.pb",
        "classes": list(range(10)), # [4,9],
        "learning_rate": 0.001,
        "nEpochs": 10,
        "batch_size": 30,
        "max_steps": 0,
        "n_test_samples": 5000,
    }

    #audio_settings = {
    #    "dataDir": os.path.abspath("./data"),
    #    "outputDir": os.path.abspath("./build/experiment"),
    #    "modelFile": "model.pb",
    #    "classes": cfg.getGenres(),
    #    "learning_rate": 0.001,
    #    "nEpochs": 10,
    #    "batch_size": 30,
    #    "max_steps": 0,
    #    "n_test_samples": 5000,
    #}

    settings = mnist_settings # audio_settings

    settings['checkpointFile'] = os.path.join(settings['outputDir'], 'model.ckpt')

    if not os.path.exists(settings['dataDir']):
        os.makedirs(settings['dataDir'])

    dataset = MnistDataset(settings['dataDir'], settings['classes'])

    # dataset = AudioDataset(cfg)

    _, nFeatures = dataset.shape(None, flat=True)
    settings['dimensions'] = [nFeatures, 128]

    print(settings)
    print(settings['dimensions'])
    print(dataset.train_path)
    # lstm_fn = lstm(nFeatures=28*28, nClasses=len(settings['classes']))
    autoencoder_fn = autoencoder(dimensions=settings['dimensions'])
    trainer = EncoderTrainer(autoencoder_fn)
    train(settings, trainer, dataset)

if __name__ == '__main__':
    main()