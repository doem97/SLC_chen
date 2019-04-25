import keras
from keras.models import Model
from keras.layers import Add, Concatenate, Flatten, Dense, Input, Dropout, Activation, Reshape, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AtrousConvolution2D, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D
from keras.regularizers import l2

def dilated_VGG(input_shape, output_shape):
    inputs = Input(input_shape)
    pad1 = ZeroPadding2D((1, 1), input_shape=input_shape)(inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1)
    conv1 = ZeroPadding2D((1, 1))(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    pad2 = ZeroPadding2D((1, 1))(pool1)
    conv2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2)
    conv2 = ZeroPadding2D((1, 1))(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    pad3 = ZeroPadding2D((1, 1))(pool2)
    conv3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3)
    conv3 = ZeroPadding2D((1, 1))(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(conv3)
    conv3 = ZeroPadding2D((1, 1))(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    pad4 = ZeroPadding2D((1, 1))(pool3)
    conv4 = AtrousConvolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = AtrousConvolution2D(512, 3, 3, activation='relu', name='conv4_2')(conv4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = AtrousConvolution2D(512, 3, 3, activation='relu', name='conv4_3')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    pad5 = ZeroPadding2D((1, 1))(pool4)
    conv5 = AtrousConvolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = AtrousConvolution2D(512, 3, 3, activation='relu', name='conv5_2')(conv5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = AtrousConvolution2D(512, 3, 3, activation='relu', name='conv5_3')(conv5)

    out_dense = Dense(output_shape, activation = 'relu')(conv5)


    model = Model(input=inputs, output=out_dense)

    return model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, output_shape, depth=50):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        output_shape (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model