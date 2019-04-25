import os
from models import dilated_VGG, resnet_v1
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class SLCModel(object):

    model_dict = {"dilated_VGG":dilated_VGG, "resnet_v1":resnet_v1}

    def __init__(self, section):
        """ receives cf section configparser.ConfigParser()["model"]
        """
        self.loss = section.get('loss')
        self.metrics = section.get('metrics').split(',')
        self.optimizer = self.getOptimizer(section.get('optimizer'))
        self.model_name = section.get('model')
    
    def loadCheckPoint(self, model_folder):
        self.checkpoint = [ModelCheckpoint(os.path.join(model_folder, "{}".format(self.model_name)+"_{epoch:02d}-{val_loss:.2f}.hdf5"), monitor = 'val_acc', save_best_only = True, verbose = 1, period = 10)]
    
    def loadModel(self, input_shape, output_shape):
        """ should be load after init
        """
        model_selected = SLCModel.model_dict[self.model_name]
        self.model = model_selected(input_shape, output_shape)
        print(self.model.summary())
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)

    def getOptimizer(self, optimizer_param):
        """ optimizer_param is a string, and can be modified here.
        """
        if optimizer_param == "adam":
            optimizer = Adam(1e-3)
        elif optimizer_param == "rmsprop":
            optimizer = optimizer_param # 'rmsprop' can be passed directly
        else:
            raise ValueError("optimizer config {} isn't defined!".format(optimizer_param))