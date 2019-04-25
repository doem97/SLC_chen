import SkinLesionClassify

config_file = "./SLC.conf"
slc = SkinLesionClassify.SkinLesionClassify(config_file)
# slc.resizeImages()
slc.loadModel() # may pass parameter later
slc.train() # in training process the data will be loaded automatically
# slc.evaluation()
# slc.test()