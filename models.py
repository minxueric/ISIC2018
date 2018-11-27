import numpy as np
from keras.models import Model
from keras.layers import merge, Flatten, Dense, Input, Dropout, Activation, Reshape
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.noise import GaussianNoise
import h5py
np.random.seed(4)

VGG16_WEIGHTS_NOTOP = '/home/ubuntu/isic/pretrained_weights/vgg16_weights.h5'
# VGG16_WEIGHTS_NOTOP = 'pretrained_weights/vgg16_weights_th_dim_ordering_th_kernels.h5'
# download .h5 weights from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# keras2.1.6 + theano

def Unet(img_rows, img_cols, loss , optimizer, metrics, fc_size = 8192, channels = 3):
    filter_size = 5
    filter_size_2 = 11
    dropout_a = 0.5
    dropout_b = 0.5
    dropout_c = 0.5
    gaussian_noise_std = 0.025

    inputs = Input((channels, img_rows, img_cols))
    input_with_noise = GaussianNoise(gaussian_noise_std)(inputs)

    conv1 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(input_with_noise)
    conv1 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv1)
    conv1 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    pool1 = GaussianNoise(gaussian_noise_std)(pool1)

    conv2 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv2)
    conv2 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    pool2 = GaussianNoise(gaussian_noise_std)(pool2)

    conv3 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(conv3)
    conv3 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(dropout_a)(pool3)

    fc = Flatten()(pool3)
    fc = Dense(fc_size, activation='relu')(fc)
    fc = Dropout(dropout_b)(fc)

    n = img_rows/2/2/2
    fc = Dense(128*n*n, activation='relu')(fc)
    fc = GaussianNoise(gaussian_noise_std)(fc)
    fc = Reshape((128,n,n))(fc)

    up1 = merge([UpSampling2D(size=(2, 2))(fc), conv3], mode='concat', concat_axis=1)
    up1 = Dropout(dropout_c)(up1)

    conv4 = Convolution2D(128, filter_size_2, filter_size_2, activation='relu', border_mode='same')(up1)
    conv4 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(conv4)
    conv4 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv4)

    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv2], mode='concat', concat_axis=1)
    up2 = Dropout(dropout_c)(up2)

    conv5 = Convolution2D(64, filter_size_2, filter_size_2, activation='relu', border_mode='same')(up2)
    conv5 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv5)
    conv5 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv5)

    up3 = merge([UpSampling2D(size=(2, 2))(conv5), conv1], mode='concat', concat_axis=1)
    up3 = Dropout(dropout_c)(up3)

    conv6 = Convolution2D(32, filter_size_2, filter_size_2, activation='relu', border_mode='same')(up3)
    conv6 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv6)
    conv6 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv6)

    conv7 = Convolution2D(1, 1, 1, activation='sigmoid')(conv6)

    model = Model(input=inputs, output=conv7)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()

    return model

# need to use keras==1.2.0
def VGG16_Unet(img_rows, img_cols, pretrained, freeze_pretrained, loss , optimizer, metrics, channels=3):
    inputs = Input((channels, img_rows, img_cols))
    pad1 = ZeroPadding2D((1, 1), input_shape=(channels, img_rows, img_cols))(inputs)
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
    conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(conv4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    pad5 = ZeroPadding2D((1, 1))(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(conv5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(conv5)

    model = Model(input=inputs, output=conv5)
    # load weights
    if pretrained:
        weights_path = VGG16_WEIGHTS_NOTOP
        f = h5py.File(weights_path)
        print (f)
        for k in range(f.attrs['nb_layers']):
            if k >= (len(model.layers) - 1):
                # ignore the last layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k+1].set_weights(weights)
        f.close()
        print('ImageNet pre-trained weights loaded.')

        if freeze_pretrained:
            for layer in model.layers:
                layer.trainable = False

    dropout_val = 0.5
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = Dropout(dropout_val)(up6)

    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Dropout(dropout_val)(up7)

    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Dropout(dropout_val)(up8)

    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Dropout(dropout_val)(up9)

    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()

    return model

def Unet2(img_rows, img_cols, loss , optimizer, metrics, fc_size = 0, channels = 3):
    filter_size = 5
    filter_size_2 = 11
    dropout_a = 0.5
    dropout_b = 0.5
    dropout_c = 0.5
    gaussian_noise_std = 0.025

    inputs = Input((channels, img_rows, img_cols))
    input_with_noise = GaussianNoise(gaussian_noise_std)(inputs)

    conv1 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(input_with_noise)
    conv1 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv1)
    conv1 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    pool1 = GaussianNoise(gaussian_noise_std)(pool1)

    conv2 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv2)
    conv2 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    pool2 = GaussianNoise(gaussian_noise_std)(pool2)

    conv3 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(conv3)
    conv3 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(dropout_a)(pool3)
    if fc_size>0:
        fc = Flatten()(pool3)
        fc = Dense(fc_size)(fc)
        fc = BatchNormalization()(fc)
        fc = Activation('relu')(fc)
        fc = Dropout(dropout_b)(fc)

        n = img_rows/2/2/2
        fc = Dense(img_rows*n*n)(fc)
        fc = BatchNormalization()(fc)
        fc = Activation('relu')(fc)
        fc = GaussianNoise(gaussian_noise_std)(fc)
        fc = Reshape((128,n,n))(fc)
    else:
        fc = Convolution2D(256, filter_size, filter_size, activation='relu', border_mode='same')(pool3)
        fc = BatchNormalization()(fc)
        fc = Dropout(dropout_b)(fc)

    up1 = merge([UpSampling2D(size=(2, 2))(fc), conv3], mode='concat', concat_axis=1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(dropout_c)(up1)

    conv4 = Convolution2D(128, filter_size_2, filter_size_2, activation='relu', border_mode='same')(up1)
    conv4 = Convolution2D(128, filter_size, filter_size, activation='relu', border_mode='same')(conv4)
    conv4 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv4)

    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv2], mode='concat', concat_axis=1)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(dropout_c)(up2)

    conv5 = Convolution2D(64, filter_size_2, filter_size_2, activation='relu', border_mode='same')(up2)
    conv5 = Convolution2D(64, filter_size, filter_size, activation='relu', border_mode='same')(conv5)
    conv5 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv5)

    up3 = merge([UpSampling2D(size=(2, 2))(conv5), conv1], mode='concat', concat_axis=1)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(dropout_c)(up3)

    conv6 = Convolution2D(32, filter_size_2, filter_size_2, activation='relu', border_mode='same')(up3)
    conv6 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv6)
    conv6 = Convolution2D(32, filter_size, filter_size, activation='relu', border_mode='same')(conv6)

    conv7 = Convolution2D(1, 1, 1, activation='sigmoid')(conv6)

    model = Model(input=inputs, output=conv7)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()

    return model

# need to use tensorflow as backend
# comment this line if use keras1.2
# from keras.layers import Conv2D, Conv2DTranspose, concatenate
def Unet_basic(img_rows, img_cols, loss, optimizer, metrics, channels=3):
    inputs = Input((channels, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()

    return model

# need to use keras==1.2
def VGG16(img_rows, img_cols, pretrained, freeze_pretrained, loss , optimizer, metrics, channels=3):
    inputs = Input((channels, img_rows, img_cols))
    pad1 = ZeroPadding2D((1, 1), input_shape=(channels, img_rows, img_cols))(inputs)
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
    conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(conv4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    pad5 = ZeroPadding2D((1, 1))(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(conv5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

    model = Model(input=inputs, output=pool5)
    # load weights
    if pretrained:
        weights_path = VGG16_WEIGHTS_NOTOP
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= (len(model.layers) - 1):
                # ignore the last layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k+1].set_weights(weights)
        f.close()
        print('ImageNet pre-trained weights loaded.')

        if freeze_pretrained:
            for layer in model.layers:
                layer.trainable = False

    x = Flatten(name='flatten')(pool5)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(input=inputs, output=x)

    #print 'start compling'
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()

    return model


if __name__  == '__main__':
    from keras.optimizers import Adam
    from metrics import jacc_coef
    img_rows, img_cols = 128, 128
    loss = 'binary_crossentropy'
    optimizer = Adam(lr=1e-5)
    metrics = [jacc_coef]
    pretrained = True
    freeze_pretrained = False
    #model = Unet(img_rows, img_cols, loss , optimizer, metrics, fc_size = 4096, channels = 3)
    #print model.summary()
    model = VGG16_Unet(img_rows, img_cols, pretrained, freeze_pretrained, loss , optimizer, metrics, channels=3)
    print model.summary()
    model = Unet2(img_rows, img_cols, loss , optimizer, metrics, fc_size = 4096, channels = 3)
    print model.summary()
    metrics = ['accuracy']
    model = VGG16(img_rows, img_cols, pretrained, freeze_pretrained, loss , optimizer, metrics, channels=3)
    print model.summary()

