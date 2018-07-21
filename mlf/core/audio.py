
import numpy as np

from mlf.core.dataset import Dataset

import SigProc

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

    def exportConfig(self):
        """
        returns a minimally complete configuration for generating this dataset
        """

        procs = SigProc.newRecipeManager(self.cfg.recipe_directory,
            ingest=self.cfg.ingest_mode,
            binPath=self.cfg.ingest_binpath).getRecipe(self.cfg.featureRecipe)

        cfg = {
            "sliceSize": self.cfg.sliceSize,
            "sliceStep": self.cfg.sliceStep,
            "procs": [],
            "classes": self.cfg.getGenres(),
        }

        for proc, opts in procs:
            cfg["procs"].append({proc.__name__: opts})

        return cfg