import os
import sys
import keras
import random
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from keras.preprocessing.image import ImageDataGenerator

def read_index_file(index_file):
    """ returns index_list, index_map, class_list from 
        the index_file.
    """
    temp_csv_dataframe = pd.read_csv(index_file, header = None)
    index_list = temp_csv_dataframe[0][1:].tolist()
    label_list = temp_csv_dataframe[1][1:].tolist()
    index_map = dict(zip(index_list, label_list))
    class_list = list(set(label_list))
    class2num = {}
    for i, x in enumerate(class_list):
        class2num[x] = i
    return index_list, index_map, class2num

def resize_image(ori_folder, resize_folder, index_list, re_size):
    """resize images on index_list from ori_folder into resize_folder 
       with re_size as new size. 
       
       re_size: a tuple of (height, width)
       
       *Notice: 
        1. default ori_folder and resize_folder is existing and resize_folder
        is empty.
        2. the suffix should be set manually, and output is the same as 
        input
    """
    if os.path.exists(ori_folder):
        if not os.path.exists(resize_folder):
            print("resize_folder didn't exist!\nthe resize_folder will be created.")
            os.makedirs(resize_folder)
        else:
            print("resize_folder already exists, any existed image in the folder will be re-write.")
    else:
        raise SystemExit("ori_folder doesn't exists!")
    print("resize the original images into resize folder {}".format(resize_folder))
    for img_index in tqdm(index_list):
        img_filename = img_index + ".jpg"
        ori_image = cv.imread(os.path.join(ori_folder, img_filename))
        resize_image = cv.resize(ori_image, re_size, interpolation=cv.INTER_CUBIC)
        cv.imwrite(os.path.join(resize_folder, img_filename), resize_image)

def split_index_list(index_list, split_ratio):
    """ index_list is a list of all index, split_ratio is [train, val, test] ratio.
    """
    random.seed(4)
    random.shuffle(index_list)
    split_ratio = [int(split_ratio[0]),int(split_ratio[1]),int(split_ratio[2])]
    train_lenth = int((split_ratio[0]/np.sum(split_ratio))*len(index_list))
    val_lenth = int((len(index_list) - train_lenth)*(split_ratio[1]/(split_ratio[1] + split_ratio[2])))
    test_lenth = (len(index_list) - train_lenth - val_lenth)
    train_list = index_list[:train_lenth]
    val_list = index_list[train_lenth:train_lenth + val_lenth]
    test_list = index_list[-test_lenth:]
    return train_list, val_list, test_list

def load_image(image_folder, index_list, height, width):
    """ index_list.jpg(or other suffix) can will be returned.
    """
    image_array = np.zeros((len(index_list), height, width, 3), dtype = np.float32) # channels_last in Keras
    for i,x in enumerate(tqdm(index_list)):
        image_path = os.path.join(image_folder, x + ".jpg")
        img = cv.imread(image_path)
        if img is None:
            raise SystemExit("the image {} cannot find!".format(image_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
        image_array[i] = img
    return image_array

def load_ctg_label(index_list, index_map, class2num):
    label_array = np.zeros((len(index_list)), dtype = np.int8)
    for i, x in enumerate(index_list):
        label_array[i] = class2num[index_map[x]]
    label_array = keras.utils.to_categorical(label_array)
    return label_array

def create_dict_from_section(section):
    """ very naive reading method Hahaha.
        read carefully!
        section is a section: configparser.ConfigParser['section_name']
    """
    args_dict = {}
    for feature in section:
        if 'False' in section[feature] or 'True' in section[feature]:
            args_dict[feature] = section.getboolean(feature)
        elif '.' in section[feature]:
            args_dict[feature] = section.getfloat(feature)
        elif 'None' in section[feature]:
            args_dict[feature] = None
        else:
            args_dict[feature] = section.get(feature)
    return args_dict

def construct_data_gen_from_dict(data_gen_args_dict):
    return ImageDataGenerator(**data_gen_args_dict)

class DataPath(object):
    """ designed for control all pathes when resize & training.
        fields:
            data_path, ori_folder, resize_folder
    """

    def __init__(self, root_path):
        self.data_path = os.path.join(root_path, "dataset")
        self.model_path = os.path.join(root_path, "model")
        self.ori_folder = os.path.join(self.data_path, "origin")
    
    def setResizeFolder(self, height, width):
        self.resize_folder = os.path.join(self.data_path, "resized_{}_{}".format(height, width))

    def setIndexFile(self, index_file_name):
        self.index_file = os.path.join(self.data_path, index_file_name)