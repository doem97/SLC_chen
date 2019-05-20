import keras
from keras.models import Model
from keras.layers import Dense

def dense_net_121(input_shape, output_shape, **kwargs):
    model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='max')
    output_layer = Dense(output_shape, activation='softmax', kernel_initializer='he_normal')(model.output)
    model = Model(inputs= model.input, outputs= output_layer)
    return model

def dense_net_169(input_shape, output_shape, **kwargs):
    model = keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='max')
    output_layer = Dense(output_shape, activation='softmax', kernel_initializer='he_normal')(model.output)
    model = Model(inputs= model.input, outputs= output_layer)
    return model

def dense_net_201(input_shape, output_shape, **kwargs):
    model = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='max')
    output_layer = Dense(output_shape, activation='softmax', kernel_initializer='he_normal')(model.output)
    model = Model(inputs= model.input, outputs= output_layer)
    return model