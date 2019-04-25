import os
import configparser
import SLC_utils
import SLC_models


class SkinLesionClassify(object):
    """ SkinLesionClassify:
        height, width, datapath
    """

    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise SystemExit("the config file {} didn't exists!".format(config_file))
        self.readConfig(config_file)
        self.iniIndexList(self.datapath.index_file)
        self.splitIndexList(self.ratio)

    def readConfig(self, config_file):
        print("loading the config file...")
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.height = cf.getint("data_profile","height")
        self.width = cf.getint("data_profile","width")
        self.input_shape = (self.height, self.width, 3)
        self.datapath = SLC_utils.DataPath(cf.get("data_profile","root_path"))
        self.datapath.setResizeFolder(self.height, self.width)
        self.datapath.setIndexFile(cf.get("data_profile","index_file"))
        self.ratio = cf.get("data_profile","ratio").split(",")
        self.image_gen_args_dict = SLC_utils.create_dict_from_section(cf["image_gen"])
        self.readTrainArgs(cf["train_args"])
        self.model_args = cf["model"]

    def readTrainArgs(self, section):
        self.batch_size = section.getint("batch_size")
        self.initial_epoch = section.getint('initial_epoch')
        self.final_epoch = section.getint('final_epoch')

    def resizeImages(self):
        SLC_utils.resize_image(self.datapath.ori_folder, self.datapath.resize_folder, self.index_list, (self.height, self.width))

    def loadModel(self):
        """ self.slcmodel.model is the keras model
            object calling
        """
        self.slcmodel = SLC_models.SLCModel(self.model_args)
        self.slcmodel.loadCheckPoint(self.datapath.model_path)
        self.slcmodel.loadModel(self.input_shape, len(self.class2num))

    def iniIndexList(self, index_file):
        """ index_file should be .csv
            init the index_list, index_map, class_list
        """
        if not os.path.exists(index_file):
            raise SystemExit("index_file {} doesn't exist!".format(index_file))
        print("loading the index file...")
        self.index_list, self.index_map, self.class2num = SLC_utils.read_index_file(index_file)

    def splitIndexList(self, split_ratio):
        """ split index list and return three lists.
            should be used after iniIndexList
        """
        print("cutting the index list...")
        self.train_list, self.val_list, self.test_list = SLC_utils.split_index_list(self.index_list, split_ratio)

    def constructDataFlow(self, data, label):
        image_datagen = SLC_utils.construct_data_gen_from_dict(self.image_gen_args_dict)
        image_datagen.fit(data, augment = True)
        train_data_flow = image_datagen.flow(data, label, batch_size = self.batch_size) # batch_size should only show up here in data augmentation situation
        return train_data_flow
    
    def getData(self, index_list):
        image = SLC_utils.load_image(self.datapath.resize_folder, index_list, self.height, self.width)
        label = SLC_utils.load_ctg_label(index_list, self.index_map, self.class2num)
        return image, label

    def train(self):
        print("begin to load training data...")
        train_image, train_label = self.getData(self.train_list)
        print("begin to load validation data...")
        val_image, val_label = self.getData(self.val_list)
        train_data_flow = self.constructDataFlow(train_image, train_label)
        history = self.slcmodel.model.fit_generator(train_data_flow, 
        validation_data = (val_image, val_label), 
        steps_per_epoch = 1, 
        initial_epoch = self.initial_epoch, 
        epochs = self.final_epoch, 
        callbacks = self.slcmodel.checkpoint,
        workers = 4,
        verbose = 1)

    def evaluation():
        return 0
    
    def test():
        return 0