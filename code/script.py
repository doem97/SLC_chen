import SkinLesionClassify
import SLC_models

config_file = "./SLC.conf"
slc = SkinLesionClassify.SkinLesionClassify(config_file)
# slc.resizeImages()
slc_model = SLC_models.SLCModel(config_file)
slc_model.loadModel() # default model in config_file will be loaded.
slc.train(slc_model)
slc_model.loadModel("dilated_VGG")
slc.train(slc_model) # in training process the data will be loaded automatically
# slc.evaluation()
# slc.test()