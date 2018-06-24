#! cd ../.. && python -m mlf.core.feature_extractor


"""

The frontend serves to prepare the audio input for training.
The code in this file controls the creation of the training
and test sets. Features are extracted and stored in HDF5 tables.
When possible, the extraction is done in parallel.

"""
import codecs
import math
import os
import sys
import random
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool
from random import shuffle

import SigProc
import numpy as np

from .config import AutoEncoderConfig
from .dataset import DatasetWriter

def abspath(path, root=None):
    """
    Find the absolute path of a path given a root directory.
    :param path: An absolute or relative path
    :param root: The root of the path (defaults to os.getcwd())
        If path is relative, it is assumed to be rooted at the given root directory.
    :return: A normalized absolute file path to the given file
    """
    if root is None:
        root = os.getcwd()
    return os.path.normpath(path) \
        if os.path.isabs(path) \
        else os.path.normpath(os.path.join(root, path))

def loadDataSetList(datalst, dataroot=None):
    """
    Reads in a dataset formatted with one element per line:
        Line format: "wavid genre path"
        wavid: unique identifier for sample
        genre: label identifying which class the file belongs to
        path:  path to the training sample

    Note: File can be any format (mp3, flac, wav...) and
        could be mono or stereo.

    :param datalst:
    :param dataroot:
    :return: 2 dictionaries:
        files  :: wavid => file_path
        labels :: wavid => genre
    """
    files = {}
    labels = {}
    with codecs.open(datalst, "r", "utf-8") as rf:
        for line in rf:
            line = line.strip()
            if line:
                wavid, label, path = line.split(None, 2)
                if path[0] == "\"":
                    path = path[1:-1]
                files[wavid] = abspath(path, dataroot)
                labels[wavid] = label
    return files, labels

def createPartition(cfg, lstfile, trfile, tefile):
    """
    partition a given list file into a training and test list
    """
    if not os.path.exists(cfg.dataDir()):
        os.makedirs(cfg.dataDir())

    files, labels = loadDataSetList(lstfile, cfg.dataroot)
    label_sets = defaultdict(list)

    for wavid, lbl in labels.items():
        label_sets[lbl].append(wavid)

    test_keys = dict()
    train_keys = dict()
    for label, data in label_sets.items():
        k = int(math.ceil(len(data) * cfg.test_ratio))
        s = set(random.sample(data, k))
        test_keys[label] = s
        train_keys[label] = set(data) - s

    with codecs.open(trfile, "w", "utf-8") as wf:
        for label, data in sorted(train_keys.items()):
            for wavid in data:
                wf.write("%s %s %s\n" % (wavid, label, files[wavid]))

    with codecs.open(tefile, "w", "utf-8") as wf:
        for label, data in sorted(test_keys.items()):
            for wavid in data:
                wf.write("%s %s %s\n" % (wavid, label, files[wavid]))

def maybeCreatePartition(cfg, lstfile):
    """
    creates a parition of the given lstfile iff the
    output does not exist, returns the two partiton files

    returns (train file, test file)
    """

    trfile = os.path.join(cfg.dataDir(), "train.lst")
    tefile = os.path.join(cfg.dataDir(), "test.lst")

    if os.path.exists(trfile) and os.path.exists(tefile):
        return (trfile, tefile)

    createPartition(cfg, lstfile, trfile, tefile)

    return (trfile, tefile)

def iter_frames(X, sliceSize, sliceStep):
    """ returns an iterator which yields (start,end)
        for each frame in the input sequence X
    """
    for s in range(0, X.shape[0] - sliceSize + 1, sliceStep):
        yield s, s + sliceSize

class FeatureExtractor(object):
    """docstring for FeatureExtractor"""

    def __init__(self, cfg, procs):
        """
        format can be "png" or "dat"
        """
        super(FeatureExtractor, self).__init__()
        self.cfg = cfg
        self.procs = procs

        self.shape = None

    def getFeatures(self, filepath):
        """
        run the recipe and return the features
        """
        data = self._runSigProc(filepath)

        return self._getFeatures(data)

    def loadFeatures(self, genre, waveid):
        """
        return the features from an already extracted file
        """

        path = self._featpath(genre, waveid)
        print("loading: %s" % path)
        mat = SigProc.Matrix.fromFile(path)

        return self._getFeatures(mat.data)

    # -------------------------------------------------------------------------

    def _runSigProc(self, inFile):
        """
        run the SigProc recipe on the given input file

        recipe is expected to return a SigProc.Matrix
        """
        proc_runner = SigProc.PipelineProcessRunner(self.procs, inFile)
        # results is an array of the output from each step in the process
        # we only care about the output from the last step
        results = proc_runner.run()

        data = results[-1]  # get the process output

        return data

    def _getFeatures(self, data):
        """
        slice an Nxd array into individual frames for training
        """
        width, height = data.shape
        sliceSize = self.cfg.sliceSize
        sliceStep = self.cfg.sliceStep

        # pad the data to fulfill the last frame
        # using the minimum value since zero may not be the min
        # vmin = np.min(data)
        # padding = vmin*np.ones( (sliceStep-width%sliceStep,height) , dtype=data.dtype)
        # data = np.hstack((data,padding))

        samples = []

        # slice the input into frames
        # note: drops the last few vectors if they do not form a complete frame
        for s, e in iter_frames(data, self.cfg.sliceSize, self.cfg.sliceStep):
            sl = data[s:e]
            if self.shape is not None:
                sl = sl.reshape(*self.shape)
            samples.append(sl)

        return np.array(samples)

    def _basename(self, waveid):
        return "%s.mat" % (waveid)

    def _basepath(self, genre):
        return os.path.join(self.cfg.sliceDir(), self.cfg.featureRecipe, genre)

    def _featpath(self, genre, waveid):
        path = self._basepath(genre)
        name = self._basename(waveid)
        return os.path.join(path, name)

    def _output_exists(self, genre, waveid):

        path = self._featpath(genre, waveid)

        try:
            return os.stat(path).st_size > 0
        except OSError:
            return False

    def extract_file(self, arg):

        waveid, genre, inFile = arg
        SigProc.Logger().register()  # this is needed for parallel pool
        # so we can print on windows

        data = self._runSigProc(inFile)
        path = self._featpath(genre, waveid)

        _, height = data.shape

        print("saving: %s" % path)
        data.toFile(path)

        # yet another hack, required to return number of frames,
        n = len(list(iter_frames(data, self.cfg.sliceSize, self.cfg.sliceStep)))

        return height, n

    def extract(self, lstfile, tag):
        """
        lstfile: a listing file (waveid, genre, filepath)
        tag: "train" or "test" or a unique name for the lstfile
        """

        genres = set()
        files, labels = loadDataSetList(lstfile, self.cfg.dataroot)
        jobs = []
        for waveid, path in files.items():
            fileGenre = labels[waveid]

            if not os.path.exists(path):
                raise Exception("Not Found: %s" % path)

            if fileGenre not in genres:
                base = self._basepath(fileGenre)

                if not os.path.exists(base):
                    os.makedirs(base)

                genres.add(fileGenre)

            if not self._output_exists(fileGenre, waveid):
                jobs.append((waveid, fileGenre, path))

        if len(jobs) > 0:
            p = Pool(self.cfg.NPARALLEL)
            results = p.map(self.extract_file, jobs)

            h, _ = zip(*results)
            # this is sort of a hack, get the height of the frame
            # each job returns the height of the frames it generated
            height = max(h)  # all values should be the same

            self._write_feats_file(self.cfg.sliceSize, height)

            if len(jobs) == len(files):
                self._write_meta(tag, genres, jobs, results)

        # save the genre labels in sorted order for training.
        path = os.path.join(self.cfg.baseDir(), "genres")
        if not os.path.exists(path):
            with codecs.open(path, "w", "utf-8") as wf:
                for g in sorted(list(genres)):
                    wf.write("%s\n" % g)

    def _write_feats_file(self, w, h):
        # we need to save the height so that i don't need to hardcode it
        # inside the config
        base = self.cfg.baseDir()
        path = os.path.join(self.cfg.baseDir(), self.cfg.featureRecipe + ".featureSize")
        if not os.path.exists(path):
            with open(path, "w") as wf:
                wf.write("%d\n" % w)
                wf.write("%d\n" % h)

    def _write_meta(self, tag, genres, jobs, results):

        # collect the number of samples per genre that are extracted
        genre_samples = defaultdict(int)
        for job, res in zip(jobs, results):
            path, waveid, genre = job
            h, n = res
            genre_samples[genre] += n

        path = os.path.join(self.cfg.baseDir(), "genres.%s.meta" % tag)
        with codecs.open(path, "w", "utf-8") as wf:
            for genre, count in sorted(genre_samples.items()):
                wf.write("%s\t%d\n" % (genre, count))

    def _loadGenre(self, genre, wavlst, framesPerFile):

        groups = [[] for i in range(len(wavlst)//4)]
        for i, waveid in enumerate(wavlst):
            groups[i%len(groups)].append(waveid)

        print("loading genre: %s nfiles: %d" % (genre, len(wavlst)))

        for group in groups:

            sample_sets = {}
            for waveid in group:
                samples = self.loadFeatures(genre, waveid)
                if samples is None:
                    raise Exception("samples not found for: %s" % waveid)

                # randomly select N samples per file
                shuffle(samples)
                samples = samples[:framesPerFile]

                sample_sets[waveid] = samples

            sample_counter = {waveid:0 for waveid in group}

            while True:

                complete = 0
                for waveid, samples in sample_sets.items():
                    if sample_counter[waveid] < len(samples):
                        yield waveid, samples[sample_counter[waveid]]
                        sample_counter[waveid] += 1
                    else:
                        complete += 1

                if complete == len(sample_sets):
                    break;

    def load_v2(self, lstfile, framesPerGenre, fptr_saver):

        files, genres = loadDataSetList(lstfile, self.cfg.dataroot)

        all_genres = self.cfg.getGenres()

        # organize files by genre
        filesByGenre = defaultdict(list)
        for wavid, genre in genres.items():
            filesByGenre[genre].append(wavid)

        percent_extra = 1.1

        for genre, wavlst in filesByGenre.items():

            framesPerFile = int(percent_extra * framesPerGenre // len(wavlst))
            label = np.array([1. if genre == g else 0. for g in all_genres])

            frames = self._loadGenre(genre, wavlst, framesPerFile)

            nframes = fptr_saver(genre, frames, label)
            print("nframes %s %d" % (genre, nframes))


def save_train_dev(cfg, ntrain, ndev, genre, frames, label):

    train_path = cfg.getDatasetPath("train", genre)
    dev_path = cfg.getDatasetPath("dev", genre)

    ctrain = 0
    cdev = 0
    ctotal = 0

    with DatasetWriter(train_path) as fTrain, DatasetWriter(dev_path) as fDev:

        try:
            while ctrain < ntrain or cdev < ndev:

                if ctrain < ntrain:
                    waveid, x = next(frames)
                    ctotal += 1
                    ctrain += 1
                    fTrain.addGrayscaleImage(x.transpose(), label, uid=waveid)

                if cdev < ndev:
                    waveid, x = next(frames)
                    ctotal += 1
                    cdev += 1
                    fDev.addGrayscaleImage(x.transpose(), label, uid=waveid)

            for x in frames:
                ctotal += 1
        except StopIteration:
            pass

    print("saved %d/%d samples for %s to %s" % (ctrain, ntrain, genre, train_path))
    print("saved %d/%d samples for %s to %s" % (cdev, ndev, genre, dev_path))
    return ctotal

def save_test(cfg, ntest, genre, frames, label):

    test_path = cfg.getDatasetPath("test", genre)

    ctest = 0
    ctotal = 0

    with DatasetWriter(test_path) as fTest:
        try:
            while ctest < ntest:

                waveid, x = next(frames)
                ctotal += 1
                ctest += 1
                fTest.addGrayscaleImage(x.transpose(), label, uid=waveid)

            for x in frames:
                ctotal += 1
        except StopIteration:
            pass

    print("saved %d/%d samples for %s to %s" % (ctest, ntest, genre, test_path))
    return ctotal

def _create_train(cfg, fe, trfile):

    print("loading training samples (%d samples per genre)" % cfg.samplesPerGenre)

    ntrain = cfg.samplesPerGenre
    ndev = int(cfg.samplesPerGenre * cfg.dev_ratio)
    framesPerGenre = ndev + ntrain

    if not os.path.exists(cfg.dataSetDir()):
        os.makedirs(cfg.dataSetDir())

    saver = lambda g, f, l: save_train_dev(cfg, ntrain, ndev, g, f, l)
    fe.load_v2(trfile, framesPerGenre, saver)

def _create_test(cfg, fe, tefile):

    ntest = int(cfg.samplesPerGenre * cfg.dev_ratio)
    framesPerGenre = ntest
    if not os.path.exists(cfg.dataSetDir()):
        os.makedirs(cfg.dataSetDir())

    saver = lambda g, f, l: save_test(cfg, ntest, g, f, l)
    fe.load_v2(tefile, framesPerGenre, saver)


def main():
    cfg = AutoEncoderConfig()
    cfg.load("./config/audio_10way.cfg")

    procs = SigProc.newRecipeManager(cfg.recipe_directory,
        ingest=cfg.ingest_mode,
        binPath=cfg.ingest_binpath).getRecipe(cfg.featureRecipe)

    fe = FeatureExtractor(cfg, procs)

    trfile, tefile = maybeCreatePartition(cfg, cfg.listfile)

    # extracting in two phases now, so that each can write
    # out a meta file with the number of frames
    fe.extract(trfile, "train")
    fe.extract(tefile, "test")

    _create_train(cfg, fe, trfile)
    _create_test(cfg, fe, tefile)

if __name__ == '__main__':
    main()