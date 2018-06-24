import codecs
import datetime
import os

from .TypedConfig import TypedConfig

class AudioConfig(TypedConfig):
    """
    AudioConfig is a config class the captures configuration settings related
    to the manipulation of audio samples for training neural networks.
    """

    def __init__(self):
        """
        Establish the basic configuration settings for audio manipulation.
        """
        super(AudioConfig, self).__init__()

        self.EXPERIMENT_ID = "experiment"

        # number of threads for parallel execution
        self.NPARALLEL = 4

        # number of jobs to split large tasks into
        self.NJOBS = 10

        # root is the directory where the experiment will be created
        # Experiment files are found in $Root/$Experiment_Id
        self.root = "."

        # dataroot specifies a path for data files
        # paths in a .lst can be relative, to the dataroot
        self.dataroot = "."

        # featureroot specifies where to extract features to.
        # For convenience, this can be separate from the experiment directory.
        self.featureroot = "./features"
        self.listfile = ""
        self.recipe_directory = "."
        self.ingest_mode = "ffmpeg"  # or "sox"
        self.ingest_binpath = "ffmpeg"  # path to ffmpeg or sox executeable

        self.train_stop_threshold = 0.01  # if change in dev accuracy is less
        self.train_stop_count = 3  # than threshold for N counts
        # then training will be stopped early

        # get all current attributes
        base_fields = self.getUngroupedAttributes()
        self.addGroup("experiment", base_fields)
        self.updateBlacklist(base_fields)

        # featureRecipe controls how features are extracted
        self.featureRecipe = "unknown"

        # sliceStep must be less than sliceSize.
        # e.g. 64 would be 50% overlap if sliceSize is 128.
        # and  96 would be 25% overlap
        self.sliceSize = 50
        self.sliceStep = 1

        # Data set parameters:
        # After generating slices, retain this many files per genre for train/dev/test
        self.samplesPerGenre = 1000

        # Percent of training samples held out from training set for the validation set
        self.dev_ratio = 0.2

        # Percent of files held out from data set for the test set
        self.test_ratio = 0.2

        self.addGroup("features", self.getUngroupedAttributes())

        # CONSTANTS:
        # These depend on other settings and are cached by the data preparation process.
        # Use the accessor methods to get these values

        # height of the spectrogram output
        self._featureHeight = 0

        # Sorted classes from input data document
        self._genres = None

        # The formatted datetime of this experiment.
        self._ctime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    def baseDir(self):
        """
        :return: The base directory of this experiment
        """
        return os.path.join(self.root, self.EXPERIMENT_ID)

    def dataDir(self):
        """
        :return: The path at which the data for this experiment is stored.
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "data")

    def sliceDir(self):
        """
        :return: The root directory of the features
        """
        return self.featureroot

    def dataSetDir(self):
        """
        :return: The directory used to store the data set.
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "data", "dataset")

    def spectrogramDir(self):
        """
        :return: The directory used to store spectrograms.
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "data", "spectrograms")

    def modelDir(self):
        """
        :return: The directory used to store the models.
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "mdl")

    def trainingSet(self):
        """
        :return: The file containing the training set.
            Training set is further partitioned into train/dev/test using the parameters above
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "data", "train.lst")

    def testSet(self):
        """
        :return: The file containing the test set.
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "data", "test.lst")

    def logDirectory(self):
        """
        :return: The directory used to store logs for this experiment
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "logs", self.featureRecipe)

    def plotDirectory(self):
        """
        :return: The directory used to store charts.
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "plots")

    def genreSet(self):
        """
        :return: The path to a file listing genre information
            This is created during dataprep from parsing train.lst
        """
        return os.path.join(self.root, self.EXPERIMENT_ID, "data", "genre.lst")

    def isPrepared(self):
        """
        :return: Has the data set been prepared?
        """
        path = os.path.join(self.baseDir(), self.featureRecipe + ".featureSize")
        return os.path.exists(path)

    def featureHeight(self):
        """
        :return: The number of features in each time-step.
        """
        if self._featureHeight == 0:
            if not self.isPrepared():
                raise RuntimeError("dataprep has not been run")
            path = os.path.join(self.baseDir(), self.featureRecipe + ".featureSize")
            with open(path) as rf:
                # Skip the first line containing the width and sliceSize
                rf.readline()
                self._featureHeight = int(rf.readline())
        return self._featureHeight

    def getGenres(self):
        """
        :return: A sorted list of the genres in this experiment
        """
        if not self._genres:
            path = os.path.join(self.baseDir(), "genres")
            if not os.path.exists(path):
                raise RuntimeError("dataprep has not been run")
            with codecs.open(path, "r", "utf-8") as rf:
                self._genres = [line.strip() for line in rf]
        return self._genres

    def ctime_s(self):
        """
        :return: The formatted datetime of the experiment
        """
        return self._ctime

    def getDatasetName(self, baseName, index, ext="tfrecord"):
        """
            basename: one of 'train' 'test' 'dev'
        """
        x = (baseName, self.featureRecipe,
             self.samplesPerGenre,
             self.sliceSize,
             self.featureHeight(), index, ext)
        return "%s_%s_%d_%dx%d_%s.%s" % x

    def getDatasetPath(self, baseName, index):
        """
            basename: one of 'train' 'test' 'dev'
        """
        name = self.getDatasetName(baseName, index)
        return os.path.join(self.dataSetDir(), name)

    def getDatasetGlobPattern(self, baseName):

        return self.getDatasetPath(baseName, "*")


class AutoEncoderConfig(AudioConfig):
    """
    RnnConfig is a subclass of AudioConfig that contains additional information
    about the settings of an LSTM or RNN.
    """

    def __init__(self):
        """
        Establish the default configuration settings.
        """
        super(AutoEncoderConfig, self).__init__()

        # Model parameters
        self.batchSize = 30
        self.nbEpoch = 20

        # Input count is decided by the recipe
        # Output count is decided by the number of classes/frontend
        # Shape of the hidden layers defined as a list of integers
        self.hidden_layer_config = [256,]

        self.learning_rate = [0.001, 0.001]

        self.optimizer = "adam"

        self.addGroup("AutoEncoder", self.getUngroupedAttributes())
