import os
import models
import configparser
from keras.optimizers import Adam

class SLCModel(object):
    """ Control everything connected to the stastic model, including:
        losses, metrics, optimizer, model(input, output)

        it's from the view of experience, things involved in improving model should
        be controled by SLCModel(), and others should be controled by SkinLesionClassify()
    """

    def __init__(self, config_file):
        """ receives cf section configparser.ConfigParser()["model"]
        """
        cf = configparser.ConfigParser()
        cf.read(config_file)
        section = cf["model"]
        self.loss = section.get('loss')
        self.metrics = section.get('metrics').split(',')
        self.getOptimizer(section.get('optimizer'))
        self.height = cf.getint("data_profile","height")
        self.width = cf.getint("data_profile","width")
        self.input_shape = (self.height, self.width, 3)
        self.output_shape = cf.getint("data_profile","output_shape")
    
    def loadModel(self, model_name = None):
        """ define self.model by model name and compile it
        """
        if not model_name:
            raise ValueError("the model_name didn't provided!")
        exec("self.model = models.{}(self.input_shape, self.output_shape)".format(model_name))
        print(self.model.summary())
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)

    def getOptimizer(self, optimizer_param):
        """ optimizer_param is a string, and can be modified here.
        """
        if optimizer_param == "adam":
            self.optimizer = Adam(1e-3)
        elif optimizer_param == "rmsprop":
            self.optimizer = optimizer_param # 'rmsprop' can be passed directly
        else:
            raise ValueError("optimizer config {} isn't defined!".format(optimizer_param))