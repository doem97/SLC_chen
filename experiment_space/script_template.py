import sys
import os
sys.path.append(os.path.join('..','code'))
import SkinLesionClassify
import SLC_models

config_file = "./SLC_template.conf"
slc = SkinLesionClassify.SkinLesionClassify(config_file)
# slc.resizeImages()
slc_model = SLC_models.SLCModel(config_file)
slc_model.loadModel('resnet_v1', depth = 20) # default model in config_file will be loaded.
slc.train(slc_model)
slc.evaluation(slc_model)