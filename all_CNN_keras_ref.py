from __future__ import print_function
from __future__ import division
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from shuffle import *
from next_batch import *
from LSUV import *

# import pandas
# import cv2
import numpy as np

K.set_image_dim_ordering('tf')

batch_size = 32
nb_classes = 10
nb_epoch = 350
rows, cols = 32, 32
channels = 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(X_train.shape[1:])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, 1, 1, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(10, 1, 1, border_mode='valid'))

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
# model = make_parallel(model, 4)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# initialize the model using LSUV
training_data_shuffled, training_labels_oh_shuffled = shuffle(X_train, Y_train)
batch_xs_init = training_data_shuffled[0:batch_size]
LSUV_init(model, batch_xs_init)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)

datagen.fit(X_train)
filepath = "keras_allconv_LSUV.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                             mode='max')

callbacks_list = [checkpoint]
# Fit the model on the batches generated by datagen.flow().
history_callback = model.fit_generator(datagen.flow(X_train, Y_train,
                                                    batch_size=batch_size),
                                       steps_per_epoch=X_train.shape[0]/batch_size,
                                       epochs=nb_epoch, validation_data=(X_test, Y_test), callbacks=callbacks_list,
                                       verbose=1)

# im = cv2.resize(cv2.imread('image.jpg'), (224, 224)).astype(np.float32)
# out = model.predict(im)
# print
# np.argmax(out)
#
# pandas.DataFrame(history_callback.history).to_csv("history.csv")

model.save('keras_allconv_LSUV.h5')