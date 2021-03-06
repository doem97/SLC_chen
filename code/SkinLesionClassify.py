import os
import SLC_utils
import numpy as np
import configparser
from SLC_utils import DataPath
from time import strftime, gmtime
from keras.callbacks import ModelCheckpoint, TensorBoard


class SkinLesionClassify(object):
    """ SkinLesionClassify:
        a class that controls the process of thorough classification training.
        control: prepare data, create datageneration, test and give score

        it's from the view of experiments, things involved in improving model should
        be controled by SLCModel(), and others should be controled by SkinLesionClassify()
    """

    def __init__(self, config_file):
        """ set
        """
        if not os.path.exists(config_file):
            raise SystemExit("the config file {} didn't exists!".format(config_file))
        self.readConfig(config_file)
        self.iniIndexList(DataPath.index_file)
        self.splitIndexList(self.ratio)

    def readConfig(self, config_file):
        print("loading the config file...")
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.height = cf.getint("data_profile","height")
        self.width = cf.getint("data_profile","width")
        self.input_shape = (self.height, self.width, 3)
        DataPath.initSettings(cf.get("data_profile","root_path"))
        DataPath.setResizeFolder(self.height, self.width)
        DataPath.setIndexFile(cf.get("data_profile","index_file"))
        temp_ratio = cf.get("data_profile","ratio").split(",")
        self.ratio = [int(temp_ratio[0]),int(temp_ratio[1]),int(temp_ratio[2])]
        self.image_gen_args_dict = SLC_utils.create_dict_from_section(cf["image_gen"])
        self.readTrainArgs(cf["train_args"])
        self.model_args = cf["model"]
        self.save_models = cf.getboolean("control", "save_models")

    def readTrainArgs(self, section):
        self.batch_size = section.getint("batch_size")
        self.initial_epoch = section.getint('initial_epoch')
        self.final_epoch = section.getint('final_epoch')

    def resizeImages(self):
        """ remember to use brackets with it.
        """
        SLC_utils.resize_image(DataPath.ori_folder, DataPath.resize_folder, self.index_list, (self.height, self.width))

    def iniIndexList(self, index_file):
        """ index_file should be .csv
            init the index_list, index_map, class_list
            index_list: a list of all data's index(string)
            index_map: mapping, {index(string):label(string)}
            class2num: mapping, {label(string):categorical_index(array)}
        """
        if not os.path.exists(index_file):
            raise SystemExit("index_file {} doesn't exist!".format(index_file))
        print("loading the index file...")
        self.index_list, self.index_map, self.class2num = SLC_utils.read_index_file(index_file)

    def splitIndexList(self, split_ratio):
        """ split the list into three parts:
            train_list, val_list, test_list
            each is [index1(string), index2...]
        """
        print("cutting the index list...")
        self.train_list, self.val_list, self.test_list = SLC_utils.split_index_list(self.index_list, split_ratio)

    def constructDataFlow(self, data, label):
        image_datagen = SLC_utils.construct_data_gen_from_dict(self.image_gen_args_dict)
        image_datagen.fit(data, augment = True)
        train_data_flow = image_datagen.flow(data, label, batch_size = self.batch_size) # batch_size should only show up here in data augmentation situation
        return train_data_flow
    
    def loadCheckPoint(self, model_name):
        model_index = "{}_{}".format(model_name, strftime("%Y%m%d_%H%M%S", gmtime()))
        tensorboard_dir = os.path.join(DataPath.log_path, model_index)
        tensorboard = TensorBoard(log_dir = tensorboard_dir)
        print("tensorboard: dir will be saved into {}".format(tensorboard_dir))
        if self.save_models:
            monitor = 'val_acc'
            checkpoint = ModelCheckpoint(os.path.join(DataPath.model_path, model_index+".hdf5"), monitor = monitor, save_best_only = True, verbose = 1, period = 1)
            print("checkpoint: models will be saved into {}, monitoring variable: {}".format(DataPath.model_path, monitor))
            return [checkpoint, tensorboard]
        else:
            return [tensorboard]

    def getData(self, index_list):
        image = SLC_utils.load_image(DataPath.resize_folder, index_list, self.height, self.width)
        label = SLC_utils.load_ctg_label(index_list, self.index_map, self.class2num)
        return image, label

    def train(self, slc_model):
        """ slc_model is a SLCModel object
        """
        print("begin to load training data...")
        train_image, train_label = self.getData(self.train_list)
        print("begin to load validation data...")
        val_image, val_label = self.getData(self.val_list)
        train_data_flow = self.constructDataFlow(train_image, train_label)
        checkpoint_list = self.loadCheckPoint(slc_model.model_name)
        history = slc_model.model.fit_generator(train_data_flow, 
        validation_data = (val_image, val_label), 
        steps_per_epoch = 1, 
        initial_epoch = self.initial_epoch, 
        epochs = self.final_epoch, 
        callbacks = checkpoint_list,
        workers = 4,
        verbose = 1)

    def evaluation(self, slc_model, save_results = False):
        """ wait: save_results didn't used cause labels may be write into .csv?
            slc_model is a SLCModel object
        """ 
        test_image, test_label = self.getData(self.test_list)
        scalar = slc_model.model.evaluate(test_image, test_label, verbose = 1)
        for i in range(len(slc_model.model.metrics_names)):
            print("test {} is {}".format(slc_model.model.metrics_names[i], scalar[i]))