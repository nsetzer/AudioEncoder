

class Model(object):
    """docstring for Model"""
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.settings = self.defaultSettings()
        self.updateSettings(kwargs)

    def defaultSettings(self):
        return {}

    def updateSettings(self, kwargs):

        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value
            else:
                raise AttributeError(key)

        for key, value in self.settings.items():
            if value is None:
                raise AttributeError("%s not set" % key)

    def __call__(self, x, y, reuse=False, isTraining=False):
        """
        models should return a dictionary containing:
            "cost": output of a loss function to minimize
        models should return x, y as part of the dictionary unmodified.
        classifiers should return:
            logits, prediction, classes
        encoders should return
            x, z, y_hat, y
        output of this function is used for exporting production models

        models should be invariant to batch_size
            It may need to be an argument to this function

        reuse: whether to use variables that already exist
            the first time this function is called, set to False
            to build the training graph.

        isTraining: whether the graph that is produced will
            be used for training
        """
        raise NotImplementedError()

