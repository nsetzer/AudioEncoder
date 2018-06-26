

class Model(object):
    """docstring for Model"""
    def __init__(self):
        super(Model, self).__init__()
        self.settings = {}

    def updateSettings(self, kwargs):

        for key, value in kwargs.items():
            if key in settings:
                settings[key] = value
            else:
                raise AttributeError(key)

        for key, value in self.settings.items():
            if value is None:
                raise AttributeError("%s not set" % key)

    def __call__(self, x, y, reuse=False, isTraining=False):
        raise NotImplementedError()

